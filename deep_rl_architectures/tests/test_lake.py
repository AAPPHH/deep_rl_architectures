import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

import pyarrow as pa

import deep_rl_architectures
from deep_rl_architectures.lake import Lake, Stream


class LakeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lake = Lake(self.tmp.name, prefix="t")

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_is_worktree(self):
        Path(deep_rl_architectures.__file__).resolve().relative_to(Path(__file__).resolve().parents[2])

    def test_run_roll_close_read(self):
        schema = pa.schema([("update", pa.int64()), ("win_rate", pa.float64())])
        with self.lake.run("r1", config={"lr": 1e-3}, streams={"metrics": schema}, roll_interval_s=0) as run:
            run.log("metrics", {"update": 0, "win_rate": 0.1})
            run.flush()
            run.log("metrics", [{"update": 1, "win_rate": 0.2}, {"update": 1, "win_rate": 0.3}])
            run.flush()
            run.log("metrics", pa.table({"update": [2], "win_rate": [0.5]}, schema=schema))
            run.flush()
        d = self.lake.run_dir("r1")
        self.assertEqual(len(list(d.glob("metrics_*.parquet"))), 3)
        self.assertFalse((d / "metrics.live.arrow").exists())
        df = self.lake.read("r1", "metrics", dedup="update")
        self.assertEqual(df["update"].tolist(), [0, 1, 2])
        self.assertEqual(df["win_rate"].tolist(), [0.1, 0.3, 0.5])
        self.assertEqual(list(self.lake.read("r1", "metrics", columns=["update"]).columns), ["update"])

    def test_live_read_before_close(self):
        run = self.lake.run("r1", roll_interval_s=3600)
        run.log("m", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        run.flush()
        self.assertEqual(self.lake.read("r1", "m")["a"].tolist(), [1, 2])
        run.log("m", {"a": 3, "b": "z"})
        run.flush()
        self.assertEqual(self.lake.read("r1", "m")["a"].tolist(), [1, 2, 3])
        run.close()
        self.assertEqual(self.lake.read("r1", "m")["a"].tolist(), [1, 2, 3])

    def test_schema_inference_and_unknown_stream(self):
        with self.lake.run("r1") as run:
            run.log("auto", {"x": 1.5, "n": 3})
        df = self.lake.read("r1", "auto")
        self.assertEqual(df["n"].tolist(), [3])
        self.assertEqual(str(df["x"].dtype), "float64")

    def test_run_meta(self):
        with self.lake.run(config={"a": 1}, resume_source="ckpt.pt") as run:
            ckpt = run.ckpt_path("m.pt")
        self.assertEqual(ckpt.parent.name, "checkpoints")
        meta = json.loads((self.lake.run_dir(run.run_id) / "run_meta.json").read_text())
        self.assertTrue(meta["run_id"].startswith("t_"))
        self.assertEqual(meta["config"], {"a": 1})
        self.assertEqual(meta["resume_source"], "ckpt.pt")
        self.assertEqual(meta["package_version"], deep_rl_architectures.__version__)
        for k in ("created_at", "git", "python", "torch", "host"):
            self.assertIn(k, meta)
        self.assertEqual([m["run_id"] for m in self.lake.runs()], [run.run_id])

    def test_query_across_runs(self):
        for rid, rates in (("a", [0.1, 0.4]), ("b", [0.2, 0.9])):
            with self.lake.run(rid) as run:
                run.log("metrics", [{"win_rate": r} for r in rates])
        live = self.lake.run("c")
        live.log("metrics", {"win_rate": 0.7})
        live.flush()
        df = self.lake.query("SELECT run_id, max(win_rate) AS m FROM metrics GROUP BY 1 ORDER BY 1")
        self.assertEqual(df["run_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(df["m"].tolist(), [0.4, 0.9, 0.7])
        live.close()

    def test_dataset_write_resolve_supersede(self):
        t = pa.table({"obs": list(range(10)), "act": [0] * 10})
        p1 = self.lake.write_dataset("expert", t, meta={"source": "bot"}, shard_rows=4)
        self.assertEqual(len(list(p1.glob("shard_*.parquet"))), 3)
        t2 = pa.table({"obs": list(range(5)), "act": [1] * 5})
        self.lake.write_dataset("expert", iter([t2, t2]), meta={"source": "bot2"}, shard_rows=100)
        self.assertEqual(self.lake.versions("expert"), [1, 2])
        d = self.lake.dataset("expert")
        self.assertEqual((d["version"], d["meta"]["rows"], d["meta"]["source"]), (2, 10, "bot2"))
        self.assertEqual(self.lake.dataset("expert", 1)["meta"]["rows"], 10)
        self.assertEqual(self.lake.read_dataset("expert", 1).column("obs").to_pylist(), list(range(10)))
        self.assertEqual(self.lake.read_dataset("expert", columns=["act"]).column("act").to_pylist(), [1] * 10)
        self.assertEqual(self.lake.dataset("expert")["registry"]["source"], "bot2")
        with self.assertRaises(KeyError):
            self.lake.dataset("nope")

    def test_registry_last_wins(self):
        t = pa.table({"x": [1, 2, 3]})
        self.lake.write_dataset("d", t, version=1)
        self.lake.write_dataset("d", pa.table({"x": [1]}), version=1)
        self.assertEqual(self.lake.registry().num_rows, 2)
        rows = self.lake.registry_resolve()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rows"], 1)
        self.assertEqual(self.lake.read_dataset("d").num_rows, 1)

    def test_int_then_float_widens(self):
        r = self.lake.run("r")
        r.log("m", {"w": 0, "n": 3})
        r.flush()
        r.log("m", {"w": 0.75, "n": 1.5})
        r.close()
        df = self.lake.read("r", "m")
        self.assertEqual(df["w"].tolist(), [0.0, 0.75])
        self.assertEqual(df["n"].tolist(), [3.0, 1.5])

    def test_reused_run_id_keeps_data_and_meta(self):
        r = self.lake.run("r", config={"a": 1})
        created = r.meta["created_at"]
        r.log("m", [{"i": 0}, {"i": 1}, {"i": 2}])
        r.flush()
        r2 = self.lake.run("r", config={"a": 2})
        r2.log("m", {"i": 9})
        r2.flush()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [0, 1, 2, 9])
        meta = self.lake.runs()[0]
        self.assertEqual((meta["created_at"], meta["config"]), (created, {"a": 1}))
        self.assertEqual(meta["resumes"][0]["config"], {"a": 2})

    def test_default_run_ids_unique(self):
        ids = {self.lake.run().run_id for _ in range(3)}
        self.assertEqual(len(ids), 3)

    def test_truncated_live_file(self):
        r = self.lake.run("r", roll_interval_s=3600)
        r.log("m", [{"i": 0}, {"i": 1}, {"i": 2}])
        r.flush()
        r.log("m", [{"i": 3}, {"i": 4}, {"i": 5}])
        r.flush()
        p = self.lake.run_dir("r") / "m.live.arrow"
        p.write_bytes(p.read_bytes()[:-10])
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [0, 1, 2])

    def test_crash_without_close_readable(self):
        r = self.lake.run("r")
        r.log("m", [{"i": 0}, {"i": 1}])
        r.flush()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [0, 1])

    def test_none_then_value_and_close_safe(self):
        r = self.lake.run("r")
        r.log("a", {"x": None})
        r.flush()
        r.log("a", {"x": 1.0})
        r.log("m", {"i": 1})
        r.close()
        self.assertEqual(self.lake.read("r", "a")["x"].dropna().tolist(), [1.0])
        self.assertFalse((self.lake.run_dir("r") / "m.live.arrow").exists())

    def test_bad_batch_does_not_poison(self):
        r = self.lake.run("r")
        r.log("m", {"i": 1})
        r.flush()
        r.log("m", {"i": "abc"})
        with self.assertRaises(Exception):
            r.flush()
        r.log("m", {"i": 2})
        r.close()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [1, 2])

    def test_query_type_conflict_isolated_and_promoted(self):
        with self.lake.run("a") as a:
            a.log("w", {"w": 1})
            a.log("other", {"z": 1})
        with self.lake.run("b") as b:
            b.log("w", {"w": 0.5, "run_id": "x"})
        self.assertEqual(self.lake.query("select count(*) c from other")["c"].tolist(), [1])
        self.assertEqual(self.lake.query("select max(w) m from w")["m"].tolist(), [1.0])

    def test_extra_keys_evolve_schema(self):
        r = self.lake.run("r")
        r.log("m", {"n": 3})
        r.flush()
        r.log("m", {"n": 4, "extra": 1.0})
        r.close()
        df = self.lake.read("r", "m")
        self.assertEqual(df["n"].tolist(), [3, 4])
        self.assertEqual(df["extra"].isna().tolist(), [True, False])
        r2 = self.lake.run("r2", streams={"m": pa.schema([("n", pa.int64())])})
        r2.log("m", [{"n": 1}, {"n": 2, "e": "x"}])
        r2.close()
        self.assertIn("e", self.lake.read("r2", "m").columns)

    def test_empty_input(self):
        r = self.lake.run("r")
        r.log("m", [])
        r.flush()
        r.close()
        self.assertEqual(len(self.lake.read("r", "m")), 0)

    def test_registry_path_posix(self):
        self.lake.write_dataset("d", pa.table({"x": [1]}))
        self.assertNotIn("\\", self.lake.registry().column("path")[0].as_py())

    def test_dataset_edge_cases(self):
        self.lake.write_dataset("e", iter([]))
        self.assertEqual(self.lake.read_dataset("e").num_rows, 0)
        self.lake.write_dataset("b", pa.record_batch({"x": [1, 2]}))
        self.assertEqual(self.lake.read_dataset("b").column("x").to_pylist(), [1, 2])
        r = self.lake.run("r")
        r.log("m", pa.record_batch({"x": [1, 2]}))
        r.close()
        self.assertEqual(self.lake.read("r", "m")["x"].tolist(), [1, 2])

    def test_req1_file_names_and_legacy_runs(self):
        d = self.lake.run_dir("old")
        d.mkdir(parents=True)
        pq.write_table(pa.table({"u": [0, 1]}), str(d / "metrics_0000.parquet"))
        pq.write_table(pa.table({"u": [2]}), str(d / "metrics_0001.parquet"))
        with pa.OSFile(str(d / "metrics.live.arrow"), "wb") as sink, ipc.new_stream(sink, pa.schema([("u", pa.int64())])) as w:
            w.write_batch(pa.record_batch({"u": [3]}))
        self.assertEqual(self.lake.read("old", "metrics")["u"].tolist(), [0, 1, 2, 3])
        self.assertEqual(self.lake.query("select count(*) c from metrics")["c"].tolist(), [4])
        r = self.lake.run("new")
        r.log("metrics", {"u": 1})
        r.flush()
        self.assertTrue((self.lake.run_dir("new") / "metrics.live.arrow").exists())
        r.close()
        self.assertEqual([f.name for f in self.lake.run_dir("new").glob("*.parquet")], ["metrics_0000.parquet"])

    def test_req2_ipc_stream_and_file_fallback(self):
        d = self.lake.run_dir("f")
        d.mkdir(parents=True)
        with pa.OSFile(str(d / "m.live.arrow"), "wb") as sink, ipc.new_file(sink, pa.schema([("u", pa.int64())])) as w:
            w.write_batch(pa.record_batch({"u": [1, 2]}))
        self.assertEqual(self.lake.read("f", "m")["u"].tolist(), [1, 2])
        (d / "m.live.arrow").write_bytes(b"garbage")
        self.assertEqual(len(self.lake.read("f", "m")), 0)
        r = self.lake.run("s", roll_interval_s=3600)
        r.log("m", [{"i": 0}, {"i": 1}])
        r.flush()
        r.log("m", [{"i": 2}])
        r.flush()
        p = self.lake.run_dir("s") / "m.live.arrow"
        p.write_bytes(p.read_bytes()[:-3])
        self.assertEqual(self.lake.read("s", "m")["i"].tolist(), [0, 1])

    def test_req3_schema_evolution_across_shards_and_runs(self):
        with self.lake.run("a") as a:
            a.log("m", {"x": 1})
            a.flush()
            a.roll_interval_s = 0
            a.log("m", {"x": 2, "extra": "e"})
        with self.lake.run("b") as b:
            b.log("m", {"y": 0.5, "x": 1.5})
        df = self.lake.read("a", "m")
        self.assertEqual(df["x"].tolist(), [1, 2])
        self.assertEqual(df["extra"].isna().tolist(), [True, False])
        self.assertEqual(self.lake.query("select count(*) c, count(extra) e from m")["e"].tolist(), [1])
        d = self.lake.run_dir("c")
        d.mkdir()
        pq.write_table(pa.table({"x": [1]}), str(d / "m_0000.parquet"))
        pq.write_table(pa.table({"x": [2.5], "z": ["q"]}), str(d / "m_0001.parquet"))
        df = self.lake.read("c", "m")
        self.assertEqual(df["x"].tolist(), [1.0, 2.5])
        self.assertEqual(df["z"].isna().tolist(), [True, False])
        self.assertEqual(self.lake.query("select max(x) m from m")["m"].tolist(), [2.5])

    def test_req4_roll_in_background_no_loss_no_dup(self):
        r = self.lake.run("r", roll_interval_s=3600)
        st = r.stream("m")
        gate = threading.Event()
        orig = st.convert

        def slow(rolling, final):
            gate.wait(5)
            orig(rolling, final)

        st.convert = slow
        r.log("m", [{"i": 0}, {"i": 1}])
        r.flush()
        t0 = time.time()
        st.roll()
        self.assertLess(time.time() - t0, 1)
        d = self.lake.run_dir("r")
        self.assertEqual([f.name for f in d.glob("*.rolling")], ["m_0000.rolling"])
        self.assertFalse((d / "m.live.arrow").exists())
        r.log("m", [{"i": 2}])
        r.flush()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [0, 1, 2])
        self.assertEqual(self.lake.query("select count(*) c from m")["c"].tolist(), [3])
        gate.set()
        r.close()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [0, 1, 2])
        self.assertEqual(list(d.glob("*.rolling")), [])
        self.assertEqual(len(list(d.glob("m_*.parquet"))), 2)

    def test_req4_concurrent_readers_and_rolls(self):
        r = self.lake.run("r", roll_interval_s=0)
        seen, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                df = self.lake.read("r", "m")
                seen.append(df["i"].tolist() if len(df) else [])

        th = threading.Thread(target=reader)
        th.start()
        for i in range(60):
            r.log("m", {"i": i})
            r.flush()
        r.close()
        stop.set()
        th.join()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), list(range(60)))
        for s in seen:
            self.assertEqual(s, list(range(len(s))))

    def test_req4_stale_rolling_salvaged_on_reopen(self):
        d = self.lake.run_dir("r")
        d.mkdir(parents=True)
        with pa.OSFile(str(d / "m_0000.rolling"), "wb") as sink, ipc.new_stream(sink, pa.schema([("i", pa.int64())])) as w:
            w.write_batch(pa.record_batch({"i": [7]}))
        r = self.lake.run("r")
        r.log("m", {"i": 8})
        r.close()
        self.assertEqual(self.lake.read("r", "m")["i"].tolist(), [7, 8])
        self.assertEqual(list(d.glob("*.rolling")), [])

    def test_req5_meta_blobs(self):
        cfg = {"tree": {"a": [1, (2, 3)], "p": Path("/x/y"), "n": np.float32(1.5), "arr": np.arange(3), 5: "k"}, "s": {1, 2}}
        with self.lake.run("r", config=cfg) as run:
            run.update_meta(sim={"dt": np.int64(2), "pos": (0.0, 1.0)}, env_kwargs={"root": Path("/e")})
        meta = json.loads((self.lake.run_dir("r") / "run_meta.json").read_text())
        self.assertEqual(meta["config"]["tree"], {"a": [1, [2, 3]], "p": "/x/y", "n": 1.5, "arr": [0, 1, 2], "5": "k"})
        self.assertEqual(meta["config"]["s"], [1, 2])
        self.assertEqual(meta["sim"], {"dt": 2, "pos": [0.0, 1.0]})
        self.assertEqual(meta["env_kwargs"], {"root": "/e"})

    def test_req6_checkpoints(self):
        run = self.lake.run("r")
        self.assertEqual(run.ckpt_path(10).name, "ckpt_10.pt")
        self.assertEqual(run.ckpt_path("final").name, "final.pt")
        self.assertEqual(run.ckpt_path("best").name, "best.pt")
        self.assertEqual(run.ckpt_path(3).parent.name, "checkpoints")
        self.assertIsNone(run.latest())
        for u in (2, 10, 9):
            run.ckpt_path(u).write_bytes(b"x")
        run.ckpt_path("final").write_bytes(b"x")
        self.assertEqual([p.name for p in run.checkpoints()], ["ckpt_2.pt", "ckpt_9.pt", "ckpt_10.pt"])
        self.assertEqual(run.latest().name, "ckpt_10.pt")
        run.close()


if __name__ == "__main__":
    unittest.main()

    def test_review_nested_streams(self):
        d = self.lake.run_dir("n1") / "traj" / "gate_events"
        d.mkdir(parents=True)
        pq.write_table(pa.table({"i": [1, 2]}), str(self.lake.run_dir("n1") / "traj" / "traj_0000.parquet"))
        pq.write_table(pa.table({"g": [1]}), str(d / "gate_events_0000.parquet"))
        self.assertEqual(len(self.lake.read("n1", "traj")), 2)
        self.assertEqual(len(self.lake.read("n1", "traj/traj")), 2)
        self.assertEqual(len(self.lake.read("n1", "gate_events")), 1)
        self.assertIn("gate_events", self.lake.stream_names())
        self.assertEqual(int(self.lake.query("select count(*) c from traj").c[0]), 2)
        run = self.lake.run("n1")
        run.log("traj/traj", {"i": 3})
        run.close()
        self.assertEqual(len(self.lake.read("n1", "traj")), 3)
        self.assertEqual(len(pq.read_table(str(self.lake.run_dir("n1") / "traj" / "traj_0000.parquet"))), 2)

    def test_review_reader_never_gapped_under_rolls(self):
        run = self.lake.run("g1", roll_interval_s=0)
        stop, bad = threading.Event(), []

        def reader():
            while not stop.is_set():
                ids = sorted(self.lake.read("g1", "x")["i"].tolist()) if run.streams.get("x") else []
                if ids != list(range(len(ids))):
                    bad.append(ids)

        th = threading.Thread(target=reader)
        th.start()
        for i in range(60):
            run.log("x", {"i": i})
            run.flush()
        stop.set()
        th.join()
        run.close()
        self.assertEqual(bad, [])
        self.assertEqual(self.lake.read("g1", "x")["i"].tolist(), list(range(60)))

    def test_review_bad_row_does_not_drop_neighbours(self):
        with self.lake.run("b1") as run:
            run.log("x", {"i": 4, "s": 1})
            run.flush()
            run.log("x", [{"i": 5, "s": "str"}, {"i": 6}])
            try:
                run.flush()
            except Exception:
                pass
        self.assertIn(6, self.lake.read("b1", "x")["i"].tolist())

    def test_review_string_vs_int_across_shards(self):
        d = self.lake.run_dir("s1")
        d.mkdir(parents=True)
        pq.write_table(pa.table({"s": ["a"]}), str(d / "m_0003.parquet"))
        pq.write_table(pa.table({"s": [1]}), str(d / "m_0004.parquet"))
        self.assertEqual(sorted(self.lake.read("s1", "m")["s"].tolist()), ["1", "a"])
        self.assertEqual(len(self.lake.query("select * from m")), 2)

    def test_review_zero_byte_live_and_meta_robust(self):
        d = self.lake.run_dir("z1")
        d.mkdir(parents=True)
        (d / "metrics.live.arrow").write_bytes(b"")
        run = self.lake.run("z1")
        run.log("metrics", {"a": 1})
        run.close()
        self.assertEqual([f.name for f in d.glob("metrics_*")], ["metrics_0000.parquet"])
        run.update_meta(v=float("nan"), w=np.float32("inf"))
        self.assertIsNone(json.loads((d / "run_meta.json").read_text())["v"])
        cyc = {}
        cyc["self"] = cyc
        run.update_meta(c=cyc)
        run.update_meta(ok=1)
        self.assertEqual(json.loads((d / "run_meta.json").read_text())["ok"], 1)

    def test_review_exit_does_not_mask_training_error(self):
        with self.assertRaises(ValueError):
            with self.lake.run("e1") as run:
                run.log("x", {"a": 1})
                run.streams["x"].convert = lambda *a: (_ for _ in ()).throw(OSError("boom"))
                raise ValueError("train")
