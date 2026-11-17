import hashlib
import json
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

REGISTRY_SCHEMA = pa.schema([("name", pa.string()), ("version", pa.int64()), ("path", pa.string()), ("rows", pa.int64()),
                             ("schema_fp", pa.string()), ("source", pa.string()), ("created_at", pa.float64())])
PARQUET_OPTS = {"compression": "zstd"}
PART = re.compile(r"^(?P<name>.+)_(?P<idx>\d{4})\.parquet$")


def to_table(item, schema=None):
    if isinstance(item, pa.Table):
        return item if schema is None or item.schema.equals(schema) else item.select(schema.names).cast(schema)
    if isinstance(item, dict):
        item = [item]
    if not isinstance(item, list):
        item = pa.Table.from_pandas(item, preserve_index=False)
        return to_table(item, schema)
    return pa.Table.from_pylist(item, schema=schema)


def schema_fingerprint(schema):
    return hashlib.sha1(";".join(f"{f.name}:{f.type}" for f in schema).encode()).hexdigest()[:16]


def read_ipc(path):
    for op in (ipc.open_stream, ipc.open_file):
        try:
            return op(str(path)).read_all()
        except Exception:
            pass
    return None


def read_parts(directory, stream, columns=None):
    directory = Path(directory)
    parts = []
    for f in sorted(directory.glob(f"{stream}_*.parquet")):
        m = PART.match(f.name)
        if m and m["name"] == stream:
            parts.append(pq.read_table(f, columns=columns))
    live = directory / f"{stream}.live.arrow"
    if live.exists():
        t = read_ipc(live)
        if t is not None and t.num_rows:
            parts.append(t.select(columns) if columns else t)
    return parts


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
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
        self.roll_idx = 1 + max([int(m["idx"]) for f in self.base_dir.glob(f"{name}_*.parquet") if (m := PART.match(f.name)) and m["name"] == name], default=-1)
        self.sink = None
        self.writer = None

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

    def write_pending(self):
        if not self.pending:
            return
        tables, rows = [], []
        for it in self.pending:
            if isinstance(it, list) or isinstance(it, dict):
                rows.extend(it if isinstance(it, list) else [it])
                continue
            if rows:
                tables.append(to_table(rows, self.schema))
                rows = []
            tables.append(to_table(it, self.schema))
            self.schema = self.schema or tables[-1].schema
        if rows:
            tables.append(to_table(rows, self.schema))
        self.pending = []
        self.schema = self.schema or tables[0].schema
        if self.writer is None:
            self.open_writer()
        for t in tables:
            t = t if t.schema.equals(self.schema) else t.select(self.schema.names).cast(self.schema)
            for b in t.to_batches():
                self.writer.write_batch(b)

    def roll(self):
        self.write_pending()
        self.last_roll = time.time()
        if self.writer is None:
            return
        self.close_writer()
        t = read_ipc(self.arrow_path)
        if t is not None and t.num_rows:
            pq.write_table(t, str(self.base_dir / f"{self.name}_{self.roll_idx:04d}.parquet"), **self.parquet_opts)
            self.roll_idx += 1
        self.arrow_path.unlink(missing_ok=True)

    def flush(self):
        self.write_pending()
        if time.time() - self.last_roll >= self.roll_interval_s:
            self.roll()

    def close(self):
        self.roll()


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
        self.meta = {"run_id": run_id, "created_at": time.time(), "config": config or {}, "git": git_hash(),
                     "package_version": module_version("deep_rl_architectures"), "python": platform.python_version(),
                     "torch": module_version("torch"), "host": socket.gethostname()}
        if resume_source is not None:
            self.meta["resume_source"] = resume_source
        atomic_write_json(self.run_dir / "run_meta.json", self.meta)

    def stream(self, name, schema=None):
        if name not in self.streams:
            self.streams[name] = Stream(self.run_dir, name, schema, self.roll_interval_s)
        return self.streams[name]

    def log(self, stream, data):
        self.stream(stream).add(data)

    def flush(self):
        for s in self.streams.values():
            s.flush()

    def close(self):
        for s in self.streams.values():
            s.close()

    def ckpt_path(self, name):
        return self.ckpt_dir / name

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Lake:
    def __init__(self, root="lake", prefix="run"):
        self.root = Path(root)
        self.prefix = prefix
        for sub in ("runs", "datasets", "recon"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.parquet"

    def run(self, run_id=None, config=None, streams=None, roll_interval_s=600, resume_source=None):
        return Run(self, run_id or f"{self.prefix}_{time.strftime('%Y%m%d_%H%M%S')}", config, streams, roll_interval_s, resume_source)

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
        parts = [t.to_pandas() for t in read_parts(self.run_dir(run_id), stream, columns)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns or [])
        if dedup and len(df):
            df = df.drop_duplicates(dedup, keep="last").reset_index(drop=True)
        return df

    def stream_names(self):
        names = set()
        for f in (self.root / "runs").glob("*/*"):
            if f.name.endswith(".live.arrow"):
                names.add(f.name[:-len(".live.arrow")])
            elif f.suffix == ".parquet" and (m := PART.match(f.name)):
                names.add(m["name"])
        return sorted(names)

    def query(self, sql):
        import duckdb
        con = duckdb.connect()
        for name in self.stream_names():
            tables = []
            for d in sorted((self.root / "runs").iterdir()):
                for t in read_parts(d, name):
                    tables.append(t.append_column("run_id", pa.array([d.name] * t.num_rows, pa.string())))
            if not tables:
                continue
            merged = pa.concat_tables(tables, promote_options="default")
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
        items = [data] if isinstance(data, (pa.Table, dict)) or hasattr(data, "to_dict") else data
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
            t = to_table(it, state["schema"])
            state["schema"] = state["schema"] or t.schema
            state["buf"].append(t)
            state["n"] += t.num_rows
            if state["n"] >= shard_rows:
                drain(False)
        drain(True)
        schema = state["schema"] or pa.schema([])
        full = {**(meta or {}), "name": name, "version": version, "rows": state["rows"], "shards": state["shards"],
                "schema": {f.name: str(f.type) for f in schema}, "created_at": time.time()}
        atomic_write_json(out / "meta.json", full)
        self.registry_append([{"name": name, "version": version, "path": str(out.relative_to(self.root)), "rows": state["rows"],
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
        return pa.concat_tables([pq.read_table(f, columns=columns) for f in sorted(path.glob("shard_*.parquet"))])
