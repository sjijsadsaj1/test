"""Preflight the multi-image capacity required by the AVIC Qwen service.

The horizon selector can receive one real anchor per frontier and the
multi-frame verifier receives one anchor plus three SVC frames per frontier.
With ApexNav's top-three frontier interface, the largest request therefore
contains twelve images.  vLLM defaults to one image per prompt, so checking
only ``/v1/models`` is not sufficient.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import zlib
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_REQUIRED_IMAGES = 12


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def make_test_png(width: int = 28, height: int = 28) -> bytes:
    """Return a small valid RGB PNG without requiring Pillow or OpenCV."""

    if width < 28 or height < 28:
        raise ValueError("Qwen test image dimensions must be at least 28x28")
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + b"\x80\x80\x80" * width
    pixels = row * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )


def build_probe_payload(model: str, image_count: int) -> Dict[str, Any]:
    if image_count < 2:
        raise ValueError("multi-image preflight requires at least two images")
    encoded = base64.b64encode(make_test_png()).decode("ascii")
    image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + encoded},
    }
    content = [
        {
            "type": "text",
            "text": "This is an AVIC multi-image capacity check. Reply with OK.",
        }
    ]
    content.extend(dict(image) for _ in range(image_count))
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 8,
    }


def check_multi_image_capacity(
    base_url: str,
    model: str,
    image_count: int = DEFAULT_REQUIRED_IMAGES,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(build_probe_payload(model, image_count)).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Qwen rejected the %d-image AVIC probe (HTTP %d): %s"
            % (image_count, exc.code, detail[:800])
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Qwen multi-image probe failed: %s" % exc) from exc
    if not isinstance(result, dict) or not result.get("choices"):
        raise RuntimeError("Qwen multi-image probe returned no choices: %r" % result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", type=int, default=DEFAULT_REQUIRED_IMAGES)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        check_multi_image_capacity(
            args.base_url,
            args.model,
            image_count=args.images,
            timeout_s=args.timeout,
        )
    except Exception as exc:
        print("AVIC Qwen multi-image preflight failed: %s" % exc, file=sys.stderr)
        return 1
    print("AVIC Qwen multi-image preflight passed (%d images)." % args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
