import hashlib
import json
import math
import numpy as np
import platform
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

REGISTRY_SCHEMA = pa.schema([("name", pa.string()), ("version", pa.int64()), ("path", pa.string()), ("rows", pa.int64()),
                             ("schema_fp", pa.string()), ("source", pa.string()), ("created_at", pa.float64())])
PARQUET_OPTS = {"compression": "zstd"}
PART = re.compile(r"^(?P<name>.+)_(?P<idx>\d{4})\.parquet$")
ROLLING = re.compile(r"^(?P<name>.+)_(?P<idx>\d{4})\.rolling$")
CKPT = re.compile(r"^ckpt_(?P<idx>\d+)\.pt$")


def rows_to_table(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return pa.table({k: [r.get(k) for r in rows] for k in keys})


def to_table(item):
    if isinstance(item, pa.Table):
        return item
    if isinstance(item, pa.RecordBatch):
        return pa.Table.from_batches([item])
    if isinstance(item, dict):
        item = [item]
    if isinstance(item, list):
        return rows_to_table(item)
    return pa.Table.from_pandas(item, preserve_index=False)


def evolve(schema, t, strict=True):
    fields = {f.name: f for f in schema}
    for f in t.schema:
        old = fields.get(f.name)
        if old is None:
            fields[f.name] = f
        elif old.type.equals(f.type) or pa.types.is_null(f.type):
            continue
        elif pa.types.is_null(old.type):
            fields[f.name] = f
        else:
            if not (pa.types.is_string(f.type) and not pa.types.is_string(old.type)):
                try:
                    t[f.name].cast(old.type)
                    continue
                except pa.ArrowException:
                    pass
            try:
                ftype = pa.unify_schemas([pa.schema([old]), pa.schema([f])], promote_options="permissive").field(0).type
            except pa.ArrowException:
                if strict:
                    raise
                ftype = pa.string()
            fields[f.name] = pa.field(f.name, ftype)
    return pa.schema(list(fields.values()))


def conform(t, schema):
    if t.schema.equals(schema):
        return t
    arrays = [t[f.name].cast(f.type) if f.name in t.column_names else pa.nulls(t.num_rows, f.type) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def concat(tables):
    tables = [t for t in tables if t.num_columns]
    if not tables:
        return None
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except pa.ArrowException:
        schema = pa.schema([])
        for t in tables:
            schema = evolve(schema, t, strict=False)
        return pa.concat_tables([conform(t, schema) for t in tables])


def schema_fingerprint(schema):
    return hashlib.sha1(";".join(f"{f.name}:{f.type}" for f in schema).encode()).hexdigest()[:16]


def read_ipc(path):
    for op in (ipc.open_stream, ipc.open_file):
        try:
            r = op(str(path))
        except Exception:
            continue
        batches = []
        try:
            if hasattr(r, "num_record_batches"):
                batches = [r.get_batch(i) for i in range(r.num_record_batches)]
            else:
                for b in r:
                    batches.append(b)
        except Exception:
            pass
        return pa.Table.from_batches(batches, schema=r.schema)
    return None


def indexed(directory, stream, pattern, suffix):
    out = {}
    path = Path(directory) / stream
    for f in path.parent.glob(f"{path.name}_*{suffix}"):
        m = pattern.match(f.name)
        if m and m["name"] == path.name:
            out[int(m["idx"])] = f
    return out


def read_part(f, columns, reader):
    t = reader(f)
    if t is None or t.num_rows == 0:
        return None
    return t if columns is None else t.select([c for c in columns if c in t.column_names])


def snapshot(directory, stream):
    return indexed(directory, stream, ROLLING, ".rolling"), indexed(directory, stream, PART, ".parquet")


def read_index(parquet, rolling, columns):
    for _ in range(2):
        if parquet.exists():
            try:
                return read_part(parquet, columns, pq.read_table)
            except FileNotFoundError:
                continue
        if rolling.exists():
            t = read_part(rolling, columns, read_ipc)
            if t is not None or rolling.exists():
                return t
    raise FileNotFoundError(parquet)


def read_parts(directory, stream, columns=None, attempts=400):
    directory = Path(directory)
    live = directory / f"{stream}.live.arrow"
    for attempt in range(attempts):
        rolling, parquet = snapshot(directory, stream)
        idxs = sorted({*rolling, *parquet})
        base = directory / stream
        try:
            parts = [read_index(Path(f"{base}_{i:04d}.parquet"), Path(f"{base}_{i:04d}.rolling"), columns) for i in idxs]
            parts.append(read_part(live, columns, read_ipc) if live.exists() else None)
        except FileNotFoundError:
            time.sleep(min(0.001 * (attempt + 1), 0.05))
            continue
        after = snapshot(directory, stream)
        if max([*after[0], *after[1]], default=-1) <= max(idxs, default=-1):
            return [p for p in parts if p is not None]
        time.sleep(min(0.001 * (attempt + 1), 0.05))
    raise RuntimeError(f"no consistent snapshot of stream {stream} in {directory}")


def stream_keys(run_dir):
    run_dir = Path(run_dir)
    keys = set()
    for f in run_dir.rglob("*"):
        rel = f.relative_to(run_dir).parent
        if f.name.endswith(".live.arrow"):
            keys.add((rel / f.name[:-len(".live.arrow")]).as_posix())
        elif f.suffix in (".parquet", ".rolling") and (m := (PART if f.suffix == ".parquet" else ROLLING).match(f.name)):
            keys.add((rel / m["name"]).as_posix())
    return sorted(keys)


def resolve_stream(run_dir, stream):
    keys = stream_keys(run_dir)
    return stream if stream in keys else next((k for k in keys if Path(k).name == stream), stream)


def jsonable(o, seen=frozenset()):
    if isinstance(o, (dict, list, tuple, set, frozenset)):
        if id(o) in seen:
            return "<cycle>"
        seen = seen | {id(o)}
    if isinstance(o, dict):
        return {k if isinstance(k, str) else str(k): jsonable(v, seen) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v, seen) for v in (sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist(), seen)
    if isinstance(o, np.generic):
        return jsonable(o.item(), seen)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe = jsonable(obj)
    with open(tmp, "w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    tmp.replace(path)


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def module_version(name):
    try:
        return getattr(__import__(name), "__version__", None)
    except Exception:
        return None


class Stream:
    def __init__(self, base_dir, name, schema=None, roll_interval_s=600, parquet_opts=None):
        self.base_dir = Path(base_dir)
        self.name = name
        self.schema = schema
        self.roll_interval_s = roll_interval_s
        self.parquet_opts = parquet_opts or PARQUET_OPTS
        self.arrow_path = self.base_dir / f"{name}.live.arrow"
        self.pending = []
        self.last_roll = time.time()
        self.executor = None
        self.futures = []
        self.sink = None
        self.writer = None
        self.arrow_path.parent.mkdir(parents=True, exist_ok=True)
        self.roll_idx = 1 + max([*indexed(self.base_dir, name, PART, ".parquet"), *indexed(self.base_dir, name, ROLLING, ".rolling")], default=-1)
        self.salvage()
        self.roll_live()

    def add(self, item):
        self.pending.append(item)

    def open_writer(self):
        self.sink = pa.OSFile(str(self.arrow_path), "wb")
        self.writer = ipc.new_stream(self.sink, self.schema)

    def close_writer(self):
        if self.writer is not None:
            self.writer.close()
            self.sink.close()
            self.writer = self.sink = None

    def append_table(self, t):
        if t.num_rows == 0:
            return
        new = t.schema if self.schema is None else evolve(self.schema, t)
        if self.schema is not None and not new.equals(self.schema):
            self.roll_live()
        self.schema = new
        if self.writer is None:
            self.open_writer()
        for b in conform(t, self.schema).to_batches():
            self.writer.write_batch(b)

    def write_pending(self):
        items, self.pending = self.pending, []
        groups = []
        for it in items:
            if isinstance(it, (list, dict)):
                rows = it if isinstance(it, list) else [it]
                if groups and isinstance(groups[-1], list):
                    groups[-1].extend(rows)
                else:
                    groups.append(list(rows))
            else:
                groups.append(it)
        errors = []
        for g in groups:
            try:
                self.append_table(to_table(g))
            except Exception as e:
                errors.append(e)
                if isinstance(g, list) and len(g) > 1:
                    for row in g:
                        try:
                            self.append_table(to_table(row))
                        except Exception as e2:
                            errors.append(e2)
        if errors:
            raise errors[0]

    def convert(self, rolling, final):
        if rolling.stat().st_size == 0:
            rolling.unlink()
            return
        t = read_ipc(rolling)
        if t is None:
            rolling.replace(rolling.with_name(f"{rolling.name}.{time.time_ns()}.corrupt"))
            return
        if t.num_rows:
            tmp = final.with_suffix(".parquet.tmp")
            pq.write_table(t, str(tmp), **self.parquet_opts)
            tmp.replace(final)
        rolling.unlink(missing_ok=True)

    def part_path(self, idx, suffix):
        return self.base_dir / f"{self.name}_{idx:04d}{suffix}"

    def salvage(self):
        for idx, f in sorted(indexed(self.base_dir, self.name, ROLLING, ".rolling").items()):
            if self.part_path(idx, ".parquet").exists():
                f.unlink(missing_ok=True)
            else:
                self.convert(f, self.part_path(idx, ".parquet"))

    def roll_live(self):
        self.close_writer()
        if not self.arrow_path.exists():
            return
        if self.arrow_path.stat().st_size == 0:
            self.arrow_path.unlink()
            return
        idx = self.roll_idx
        rolling = self.part_path(idx, ".rolling")
        self.arrow_path.replace(rolling)
        self.roll_idx += 1
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"roll-{self.name}")
        self.futures.append(self.executor.submit(self.convert, rolling, self.part_path(idx, ".parquet")))

    def join(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        futures, self.futures = self.futures, []
        for f in futures:
            f.result()

    def roll(self):
        try:
            self.write_pending()
        finally:
            self.roll_live()
            self.last_roll = time.time()

    def flush(self):
        self.write_pending()
        if time.time() - self.last_roll >= self.roll_interval_s:
            self.roll()

    def close(self):
        try:
            self.roll()
        finally:
            self.join()


class Run:
    def __init__(self, lake, run_id, config=None, streams=None, roll_interval_s=600, resume_source=None):
        self.lake = lake
        self.run_id = run_id
        self.run_dir = lake.root / "runs" / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.roll_interval_s = roll_interval_s
        self.streams = {}
        for name, schema in (streams or {}).items():
            self.stream(name, schema)
        old = {}
        try:
            old = json.loads((self.run_dir / "run_meta.json").read_text())
        except Exception:
            pass
        self.meta = {"run_id": run_id, "created_at": time.time(), "config": config or {}, "git": git_hash(),
                     "package_version": module_version("deep_rl_architectures"), "python": platform.python_version(),
                     "torch": module_version("torch"), "host": socket.gethostname()}
        if resume_source is not None:
            self.meta["resume_source"] = resume_source
        if old:
            resume = {k: self.meta[k] for k in ("created_at", "config", "git", "host") + (("resume_source",) if resume_source is not None else ())}
            self.meta = {**old, "resumes": old.get("resumes", []) + [resume]}
        atomic_write_json(self.run_dir / "run_meta.json", self.meta)

    def stream(self, name, schema=None):
        if name not in self.streams:
            self.streams[name] = Stream(self.run_dir, name, schema, self.roll_interval_s)
        return self.streams[name]

    def log(self, stream, data):
        self.stream(stream).add(data)

    def each(self, op):
        err = None
        for s in list(self.streams.values()):
            try:
                getattr(s, op)()
            except Exception as e:
                err = err or e
        if err:
            raise err

    def flush(self):
        self.each("flush")

    def close(self):
        self.each("close")

    def update_meta(self, **blobs):
        new = {**self.meta, **blobs}
        atomic_write_json(self.run_dir / "run_meta.json", new)
        self.meta = new

    def ckpt_path(self, name):
        if isinstance(name, str):
            return self.ckpt_dir / (name if "." in name else f"{name}.pt")
        return self.ckpt_dir / f"ckpt_{int(name)}.pt"

    def checkpoints(self):
        found = [(int(m["idx"]), f) for f in self.ckpt_dir.iterdir() if (m := CKPT.match(f.name))]
        return [f for _, f in sorted(found)]

    def latest(self):
        found = self.checkpoints()
        return found[-1] if found else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:
            if exc[0] is None:
                raise
        return False


class Lake:
    def __init__(self, root="lake", prefix="run"):
        self.root = Path(root)
        self.prefix = prefix
        self.registry_path = self.root / "registry.parquet"

    def run(self, run_id=None, config=None, streams=None, roll_interval_s=600, resume_source=None):
        if run_id is None:
            base, n = f"{self.prefix}_{time.strftime('%Y%m%d_%H%M%S')}", 0
            while True:
                run_id = base if n == 0 else f"{base}_{n}"
                try:
                    (self.root / "runs" / run_id).mkdir(parents=True)
                    break
                except FileExistsError:
                    n += 1
        return Run(self, run_id, config, streams, roll_interval_s, resume_source)

    def run_dir(self, run_id):
        return self.root / "runs" / run_id

    def runs(self):
        out = []
        for d in sorted((self.root / "runs").iterdir()):
            if not d.is_dir():
                continue
            try:
                out.append(json.loads((d / "run_meta.json").read_text()))
            except Exception:
                out.append({"run_id": d.name})
        return out

    def read(self, run_id, stream, columns=None, dedup=None):
        import pandas as pd
        merged = concat(read_parts(self.run_dir(run_id), resolve_stream(self.run_dir(run_id), stream), columns))
        df = merged.to_pandas() if merged is not None else pd.DataFrame(columns=columns or [])
        if dedup and len(df):
            df = df.drop_duplicates(dedup, keep="last").reset_index(drop=True)
        return df

    def stream_names(self):
        return sorted({Path(k).name for d in (self.root / "runs").iterdir() if d.is_dir() for k in stream_keys(d)})

    def query(self, sql):
        import duckdb
        con = duckdb.connect()
        for name in self.stream_names():
            if not re.search(rf"\b{re.escape(name)}\b", sql):
                continue
            tables = []
            for d in sorted((self.root / "runs").iterdir()):
                if not d.is_dir():
                    continue
                for key in (k for k in stream_keys(d) if Path(k).name == name):
                    for t in read_parts(d, key):
                        if "run_id" in t.column_names:
                            t = t.rename_columns(["stream_run_id" if c == "run_id" else c for c in t.column_names])
                        tables.append(t.append_column("run_id", pa.array([d.name] * t.num_rows, pa.string())))
            merged = concat(tables)
            if merged is None:
                continue
            con.register(f"_{name}_tbl", merged)
            con.execute(f'CREATE VIEW "{name}" AS SELECT * FROM "_{name}_tbl"')
        return con.execute(sql).df()

    def registry(self):
        return pq.read_table(self.registry_path) if self.registry_path.exists() else REGISTRY_SCHEMA.empty_table()

    def registry_append(self, rows):
        merged = pa.concat_tables([self.registry(), pa.Table.from_pylist(rows, schema=REGISTRY_SCHEMA)])
        tmp = self.registry_path.with_suffix(".parquet.tmp")
        pq.write_table(merged, str(tmp), **PARQUET_OPTS)
        tmp.replace(self.registry_path)
        return merged

    def registry_resolve(self):
        latest = {}
        for row in self.registry().to_pylist():
            latest[(row["name"], row["version"])] = row
        return list(latest.values())

    def versions(self, name):
        return sorted(r["version"] for r in self.registry_resolve() if r["name"] == name)

    def write_dataset(self, name, data, meta=None, version=None, shard_rows=1_000_000):
        version = version if version is not None else (self.versions(name) or [0])[-1] + 1
        out = self.root / "datasets" / name / f"v{version}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        items = [data] if isinstance(data, (pa.Table, pa.RecordBatch, dict)) or hasattr(data, "to_dict") else data
        state = {"schema": None, "buf": [], "n": 0, "shards": 0, "rows": 0}

        def emit(table):
            pq.write_table(table, str(out / f"shard_{state['shards']:04d}.parquet"), **PARQUET_OPTS)
            state["shards"] += 1
            state["rows"] += table.num_rows

        def drain(final):
            if not state["buf"]:
                return
            table = pa.concat_tables(state["buf"])
            state["buf"], state["n"] = [], 0
            while table.num_rows >= shard_rows:
                emit(table.slice(0, shard_rows))
                table = table.slice(shard_rows)
            if table.num_rows and final:
                emit(table)
            elif table.num_rows:
                state["buf"], state["n"] = [table], table.num_rows

        for it in items:
            t = to_table(it)
            new = t.schema if state["schema"] is None else evolve(state["schema"], t)
            if state["schema"] is not None and not new.equals(state["schema"]):
                state["buf"] = [conform(b, new) for b in state["buf"]]
            state["schema"] = new
            t = conform(t, new)
            state["buf"].append(t)
            state["n"] += t.num_rows
            if state["n"] >= shard_rows:
                drain(False)
        drain(True)
        schema = state["schema"] or pa.schema([])
        full = {**(meta or {}), "name": name, "version": version, "rows": state["rows"], "shards": state["shards"],
                "schema": {f.name: str(f.type) for f in schema}, "created_at": time.time()}
        atomic_write_json(out / "meta.json", full)
        self.registry_append([{"name": name, "version": version, "path": out.relative_to(self.root).as_posix(), "rows": state["rows"],
                               "schema_fp": schema_fingerprint(schema), "source": (meta or {}).get("source"), "created_at": full["created_at"]}])
        return out

    def dataset(self, name, version=None):
        rows = {r["version"]: r for r in self.registry_resolve() if r["name"] == name}
        if not rows:
            raise KeyError(f"dataset {name} not in registry")
        version = max(rows) if version is None else version
        if version not in rows:
            raise KeyError(f"dataset {name} version {version} not in registry")
        path = self.root / rows[version]["path"]
        return {"path": path, "version": version, "meta": json.loads((path / "meta.json").read_text()), "registry": rows[version]}

    def read_dataset(self, name, version=None, columns=None):
        path = self.dataset(name, version)["path"]
        merged = concat([pq.read_table(f, columns=columns) for f in sorted(path.glob("shard_*.parquet"))])
        return pa.table({}) if merged is None else merged
