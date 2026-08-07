from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from spatial_vlm_eval.benchmarks.spbench_si import scheduled_batch as sp_scheduled_batch
from spatial_vlm_eval.orchestration.internvl3_78b_three_bench import (
    BENCHMARK_ORDER,
    MODEL_REVISION,
    PROFILE_KEY,
    SERVED_MODEL_NAME,
    BenchmarkSpec,
    Controller,
    MultiLock,
    ResourceBlocked,
    assert_shared_profile_identity,
    baseline_errors,
    execute_workflow,
)


class _FakeProcess:
    def __init__(self, pid: int = 4321, status: int = 0) -> None:
        self.pid = pid
        self.status = status
        self.waited = False

    def poll(self) -> None:
        return None

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        self.waited = True
        return self.status


def _spec(root: Path, key: str) -> BenchmarkSpec:
    label = {"q_spatial": "Q-Spatial", "spbench_si": "SPBench-SI", "cv_bench": "CV-Bench"}[key]
    output = root / key
    track = output / "runs" / PROFILE_KEY / MODEL_REVISION / "protocol"
    return BenchmarkSpec(
        key=key,
        label=label,
        scripts_dir=root / "scripts" / key,
        output_root=output,
        profile=SimpleNamespace(inference_protocol=f"{key}-protocol", registry_digest=f"{key}-digest"),
        track=track,
        predictions=track / "predictions.jsonl",
        validation_report=track / "prediction_validation.json",
        report=output / f"{key}.md",
        total_profiles=23 if key == "cv_bench" else 21,
        official_size={"q_spatial": 271, "spbench_si": 1009, "cv_bench": 2638}[key],
        scorer_protocol=f"{key}-scorer",
    )


class ThreeBenchControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[2]
        cls.entrypoint = (
            cls.repository / "scripts" / "internvl3_78b" / "run_three_bench_evaluation.sh"
        )

    def test_three_profiles_share_exact_service_identity_revision_tp_and_processor(self) -> None:
        profiles = assert_shared_profile_identity()
        self.assertEqual(tuple(profiles), BENCHMARK_ORDER)
        for profile in profiles.values():
            self.assertEqual(profile.served_model_name, SERVED_MODEL_NAME)
            self.assertEqual(profile.revision, MODEL_REVISION)
            self.assertEqual(profile.default_tensor_parallel_size, 4)
            self.assertEqual(profile.processor_family, "internvl3")
            self.assertEqual(profile.decoding["seed"], 42)

    def test_dry_run_starts_one_service_and_preserves_q_sp_cv_order(self) -> None:
        completed = subprocess.run(
            ["bash", str(self.entrypoint), "--dry-run"],
            cwd=self.repository,
            env={
                **os.environ,
                "INTERNVL3_78B_THREE_BENCH_ENV_FILE": "/dev/null",
                "LATENT_PYTHON": sys.executable,
            },
            check=True,
            capture_output=True,
            text=True,
        )
        output = completed.stdout
        self.assertEqual(output.count("[three-bench-vllm] dry-run:"), 1)
        self.assertEqual(output.count(" vllm serve "), 1)
        positions = [output.index(label + ": bash") for label in ("Q-Spatial", "SPBench-SI", "CV-Bench")]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("background bash", output)
        self.assertIn("wait for all background publication workers", output)
        self.assertIn("no files, GPU processes, inference, scoring, or reports were changed", output)

    def test_publication_starts_before_next_inference_and_failures_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = tuple(_spec(Path(directory), key) for key in BENCHMARK_ORDER)
            events: list[str] = []
            service_started = False

            def ensure_service() -> None:
                nonlocal service_started
                if not service_started:
                    service_started = True
                    events.append("service")

            outcome = execute_workflow(
                specs,
                is_reusable=lambda _spec: False,
                ensure_service=ensure_service,
                run_inference=lambda spec: events.append(f"infer:{spec.key}"),
                start_publication=lambda spec: events.append(f"publish:{spec.key}") or spec.key,
                stop_service=lambda: events.append("stop"),
                wait_publications=lambda workers: {
                    key: 7 if key == "q_spatial" else 0 for key in workers
                },
            )
        self.assertEqual(
            events,
            [
                "service",
                "infer:q_spatial",
                "publish:q_spatial",
                "infer:spbench_si",
                "publish:spbench_si",
                "infer:cv_bench",
                "publish:cv_bench",
                "stop",
            ],
        )
        self.assertIsNone(outcome.inference_error)
        self.assertEqual(outcome.publication_status["q_spatial"], 7)
        self.assertFalse(outcome.passed)

    def test_inference_failure_stops_later_benchmarks_but_waits_prior_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = tuple(_spec(Path(directory), key) for key in BENCHMARK_ORDER)
            events: list[str] = []

            def infer(spec: BenchmarkSpec) -> None:
                events.append(f"infer:{spec.key}")
                if spec.key == "spbench_si":
                    raise RuntimeError("synthetic inference failure")

            def wait(workers: dict[str, str]) -> dict[str, int]:
                events.append("wait:" + ",".join(workers))
                return {key: 0 for key in workers}

            outcome = execute_workflow(
                specs,
                is_reusable=lambda _spec: False,
                ensure_service=lambda: None,
                run_inference=infer,
                start_publication=lambda spec: events.append(f"publish:{spec.key}") or spec.key,
                stop_service=lambda: events.append("stop"),
                wait_publications=wait,
            )
        self.assertEqual(
            events,
            [
                "infer:q_spatial",
                "publish:q_spatial",
                "infer:spbench_si",
                "stop",
                "wait:q_spatial",
            ],
        )
        self.assertIn("synthetic inference failure", outcome.inference_error or "")
        self.assertNotIn("infer:cv_bench", events)

    def test_all_reusable_predictions_skip_service_and_only_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = tuple(_spec(Path(directory), key) for key in BENCHMARK_ORDER)
            calls: list[str] = []
            outcome = execute_workflow(
                specs,
                is_reusable=lambda _spec: True,
                ensure_service=lambda: calls.append("unexpected-service"),
                run_inference=lambda spec: calls.append(f"unexpected-infer:{spec.key}"),
                start_publication=lambda spec: calls.append(f"publish:{spec.key}") or spec.key,
                stop_service=lambda: calls.append("stop"),
                wait_publications=lambda workers: {key: 0 for key in workers},
            )
        self.assertEqual(
            calls,
            ["publish:q_spatial", "publish:spbench_si", "publish:cv_bench", "stop"],
        )
        self.assertTrue(outcome.passed)

    def test_baseline_requires_exact_registry_set_or_only_internvl78_missing(self) -> None:
        from spatial_vlm_eval.benchmarks.q_spatial.profiles import PROFILE_SEQUENCE

        with tempfile.TemporaryDirectory() as directory:
            spec = _spec(Path(directory), "q_spatial")
            expected = set(PROFILE_SEQUENCE)
            without_78b = expected - {PROFILE_KEY}
            with mock.patch(
                "spatial_vlm_eval.orchestration.internvl3_78b_three_bench._report_profiles",
                return_value=without_78b,
            ):
                self.assertEqual(baseline_errors([spec]), [])
            with mock.patch(
                "spatial_vlm_eval.orchestration.internvl3_78b_three_bench._report_profiles",
                return_value=expected,
            ):
                self.assertEqual(baseline_errors([spec]), [])
            invalid = (without_78b - {next(iter(without_78b))}) | {"unexpected_profile"}
            with mock.patch(
                "spatial_vlm_eval.orchestration.internvl3_78b_three_bench._report_profiles",
                return_value=invalid,
            ):
                self.assertTrue(baseline_errors([spec]))

    def test_multi_lock_conflict_fails_closed_without_releasing_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control" / "lock"
            owner = MultiLock([path])
            owner.acquire()
            try:
                with self.assertRaises(ResourceBlocked):
                    MultiLock([path]).acquire()
                self.assertEqual(len(owner.handles), 1)
            finally:
                owner.release()

    def test_spbench_scheduler_returns_four_on_lock_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"SPBENCH_SI_OUTPUT_ROOT": directory}
        ), mock.patch.object(
            sp_scheduled_batch.fcntl,
            "flock",
            side_effect=BlockingIOError,
        ):
            status = sp_scheduled_batch.run_controller(self.repository, stage="test")
        self.assertEqual(status, 4)

    def test_cleanup_signals_only_recorded_owned_process_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = tuple(_spec(root, key) for key in BENCHMARK_ORDER)
            controller = Controller(
                self.repository,
                specs,
                root / "control",
                {
                    "INTERNVL3_78B_THREE_BENCH_STOP_TIMEOUT_SECONDS": "1",
                    "INTERNVL3_78B_THREE_BENCH_BASE_URL": "http://127.0.0.1:18103/v1",
                },
            )
            controller.prepare()
            active = _FakeProcess(pid=8765)
            service = _FakeProcess(pid=8766)
            publication = _FakeProcess(pid=8767)
            controller.active_step = active  # type: ignore[assignment]
            controller.service = service  # type: ignore[assignment]
            controller.publication_workers["q_spatial"] = publication  # type: ignore[assignment]
            with mock.patch("os.killpg") as killpg:
                controller.cleanup()
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(8765, 15), mock.call(8766, 15), mock.call(8767, 15)],
        )
        self.assertTrue(active.waited)
        self.assertTrue(service.waited)
        self.assertTrue(publication.waited)
        source = (
            self.repository
            / "src"
            / "spatial_vlm_eval"
            / "orchestration"
            / "internvl3_78b_three_bench.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)


if __name__ == "__main__":
    unittest.main()
