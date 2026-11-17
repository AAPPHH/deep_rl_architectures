import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

import deep_rl_architectures
from deep_rl_architectures.lake import Lake


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


if __name__ == "__main__":
    unittest.main()
