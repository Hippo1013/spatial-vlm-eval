import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class ServiceReadinessProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[2]
        cls.probe_path = (
            cls.repository / "scripts" / "msmu" / "_probe_openai_models.py"
        )
        spec = importlib.util.spec_from_file_location("msmu_model_probe", cls.probe_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load readiness probe: {cls.probe_path}")
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def model_is_ready(self, payload, status=200):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        response = MagicMock(status=status)
        response.read.return_value = body
        connection = MagicMock()
        connection.getresponse.return_value = response
        with patch.object(
            self.probe.http.client, "HTTPConnection", return_value=connection
        ) as connection_factory:
            ready = self.probe.model_is_ready(
                "http://127.0.0.1:18081/v1", "expected-model", 5.0
            )
        connection_factory.assert_called_once_with("127.0.0.1", 18081, timeout=5.0)
        connection.request.assert_called_once_with(
            "GET", "/v1/models", headers={"Accept": "application/json"}
        )
        connection.close.assert_called_once_with()
        return ready

    def test_accepts_an_exact_model_id(self):
        self.assertTrue(
            self.model_is_ready(
                {"object": "list", "data": [{"id": "expected-model"}]}
            )
        )

    def test_rejects_substring_malformed_and_http_error_responses(self):
        cases = [
            ({"data": [{"id": "expected-model-variant"}]}, 200),
            (b"not-json", 200),
            ({"data": [{"id": "expected-model"}]}, 503),
        ]
        for payload, status in cases:
            with self.subTest(payload=payload, status=status):
                self.assertFalse(self.model_is_ready(payload, status=status))

    def test_rejects_non_local_or_non_http_endpoints_without_connecting(self):
        with patch.object(self.probe.http.client, "HTTPConnection") as connection_factory:
            for base_url in (
                "https://127.0.0.1:18081/v1",
                "http://example.com/v1",
            ):
                with self.subTest(base_url=base_url):
                    self.assertFalse(
                        self.probe.model_is_ready(base_url, "expected-model", 5.0)
                    )
        connection_factory.assert_not_called()

    def test_network_failure_is_not_ready_and_connection_is_closed(self):
        connection = MagicMock()
        connection.request.side_effect = TimeoutError
        with patch.object(
            self.probe.http.client, "HTTPConnection", return_value=connection
        ):
            self.assertFalse(
                self.probe.model_is_ready(
                    "http://localhost:18080/v1", "expected-model", 5.0
                )
            )
        connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
