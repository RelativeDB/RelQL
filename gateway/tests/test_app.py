import base64
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

os.environ.update(CLOUDMAP_NAMESPACE="test", CLOUDMAP_SERVICE="workers",
                  KEYS_TABLE="keys", USAGE_TABLE="usage")


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules["boto3"] = types.SimpleNamespace(client=MagicMock())
        cls.boto = patch("boto3.client", return_value=MagicMock())
        cls.boto.start()
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        cls.app = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        cls.boto.stop()
        sys.modules.pop("boto3", None)

    def test_json_forward_tokens(self):
        body = json.dumps({"batch": {"b": 3, "s": 17}}).encode()
        self.assertEqual(self.app._model_tokens("/v1/forward", body), 51)

    def test_binary_forward_tokens(self):
        header = json.dumps({"b": 2, "s": 19}).encode()
        body = len(header).to_bytes(4, "little") + header + b"arrays"
        self.assertEqual(self.app._model_tokens("/v2/forward", body), 38)

    def test_binary_response_is_base64(self):
        out = self.app._response(200, b"\x00\xff", "application/octet-stream")
        self.assertTrue(out["isBase64Encoded"])
        self.assertEqual(base64.b64decode(out["body"]), b"\x00\xff")

    def test_rejects_unknown_path_before_auth(self):
        out = self.app.handler({"httpMethod": "GET", "path": "/admin"}, None)
        self.assertEqual(out["statusCode"], 404)

    def test_streaming_chat_forces_usage_chunk(self):
        body, streaming = self.app._llm_body(
            "/v1/chat/completions",
            json.dumps({"model": "gpt-5.6-luna", "stream": True}).encode())
        self.assertTrue(streaming)
        self.assertTrue(json.loads(body)["stream_options"]["include_usage"])

    def test_usage_supports_both_api_shapes(self):
        self.assertEqual(
            self.app._usage({"usage": {
                "prompt_tokens": 4, "completion_tokens": 3,
                "total_tokens": 7}}), (4, 3, 7))
        self.assertEqual(
            self.app._usage({"response": {"usage": {
                "input_tokens": 8, "output_tokens": 5,
                "total_tokens": 13}}}), (8, 5, 13))


if __name__ == "__main__":
    unittest.main()
