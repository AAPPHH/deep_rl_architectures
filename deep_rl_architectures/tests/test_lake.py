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
        self.assertTrue(Path(deep_rl_architectures.__file__).exists())

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


if __name__ == "__main__":
    unittest.main()
