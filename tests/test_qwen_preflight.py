import json
import struct
import unittest
from unittest.mock import patch

from avic_bridge.qwen_preflight import (
    DEFAULT_REQUIRED_IMAGES,
    build_probe_payload,
    check_multi_image_capacity,
    make_test_png,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class QwenMultiImagePreflightTests(unittest.TestCase):
    def test_generated_probe_image_is_valid_png(self):
        image = make_test_png()
        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(image[12:16], b"IHDR")
        width, height, depth, color_type = struct.unpack(">IIBB", image[16:26])
        self.assertEqual((width, height), (28, 28))
        self.assertEqual((depth, color_type), (8, 2))

    def test_probe_matches_largest_top_three_verifier_request(self):
        payload = build_probe_payload("test-model", DEFAULT_REQUIRED_IMAGES)
        content = payload["messages"][0]["content"]
        images = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(DEFAULT_REQUIRED_IMAGES, 12)
        self.assertEqual(len(images), DEFAULT_REQUIRED_IMAGES)

    @patch("avic_bridge.qwen_preflight.urlopen")
    def test_success_requires_chat_completion_choices(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "OK"}}]}
        )
        result = check_multi_image_capacity(
            "http://127.0.0.1:8001/v1", "test-model"
        )
        self.assertTrue(result["choices"])
        request = mocked_urlopen.call_args.args[0]
        submitted = json.loads(request.data.decode("utf-8"))
        content = submitted["messages"][0]["content"]
        self.assertEqual(
            sum(item["type"] == "image_url" for item in content),
            DEFAULT_REQUIRED_IMAGES,
        )

    @patch("avic_bridge.qwen_preflight.urlopen")
    def test_response_without_choices_fails_preflight(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse({"error": "bad deployment"})
        with self.assertRaisesRegex(RuntimeError, "returned no choices"):
            check_multi_image_capacity(
                "http://127.0.0.1:8001/v1", "test-model"
            )


if __name__ == "__main__":
    unittest.main()
