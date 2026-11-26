import base64
import binascii
import json
import math

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_ARTIFACT_BYTES = 24 * 1024 * 1024
MAX_RESPONSE_BYTES = 34 * 1024 * 1024


class DocTamperError(RuntimeError):

    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


def request_detector(config, endpoint, data=None):

    url = config["base_url"].rstrip("/") + endpoint

    request = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/octet-stream"
        },
        method="POST" if data is not None else "GET"
    )

    limit = (
        MAX_RESPONSE_BYTES
        if data is not None
        else 65536
    )

    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read(limit + 1)

        if len(raw) > limit:
            raise DocTamperError(
                "DocTamper response exceeds size limit."
            )

        result = json.loads(raw)

    except HTTPError as exc:

        messages = {
            400: "Empty or invalid image.",
            413: "Image exceeds the 10 MiB limit.",
            415: "Unsupported request format.",
            422: "Unsupported, damaged or oversized image."
        }

        if exc.code in messages:
            raise DocTamperError(
                messages[exc.code],
                exc.code
            ) from exc

        raise DocTamperError(
            "DocTamper service returned an error."
        ) from exc

    except (URLError, TimeoutError, OSError) as exc:
        raise DocTamperError(
            "DocTamper service is unavailable."
        ) from exc

    except ValueError as exc:
        raise DocTamperError(
            "Invalid DocTamper response."
        ) from exc

    if not isinstance(result, dict):
        raise DocTamperError(
            "Unexpected DocTamper response."
        )

    return result


def check_doctamper_health(config):

    result = request_detector(config, "/health")

    if (
        result.get("status") != "ok"
        or result.get("model") != "SmallUNet"
        or result.get("threshold") != 0.5
    ):
        raise DocTamperError(
            "DocTamper service is unhealthy."
        )

    actual_hash = str(
        result.get("model_sha256", "")
    ).lower()

    expected_hash = config["model_sha256"].lower()

    if actual_hash != expected_hash:
        raise DocTamperError(
            "DocTamper model integrity mismatch."
        )

    return result


def valid_number(value, minimum, maximum):

    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def scan_document(data, config):

    if not data:
        raise DocTamperError("Empty image.", 400)

    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocTamperError(
            "Image exceeds the 10 MiB limit.", 413
        )

    check_doctamper_health(config)

    result = request_detector(
        config,
        "/predict",
        data
    )

    signals = result.get("signals")

    if (
        result.get("module")
        != "document_tampering_detection"
        or result.get("model") != "SmallUNet"
        or result.get("status") != "completed"
        or not isinstance(signals, dict)
        or result.get("requires_calibration") is not True
        or "document_fraud_confirmed" not in result
        or result["document_fraud_confirmed"] is not None
        or result.get("automatic_block") is not False
    ):
        raise DocTamperError(
            "Unexpected DocTamper prediction or policy."
        )

    if (
        not valid_number(
            signals.get("mean_pixel_probability"), 0, 1
        )
        or not valid_number(
            signals.get("tampered_region_percentage"), 0, 100
        )
        or signals.get("threshold") != 0.5
        or signals.get("mask_available") is not True
    ):
        raise DocTamperError(
            "Invalid DocTamper numerical signals."
        )

    if (
        result.get("artifact_encoding") != "base64"
        or result.get("artifact_content_type") != "image/png"
    ):
        raise DocTamperError(
            "Unexpected artifact encoding."
        )

    artifacts = result.get("artifacts")

    expected_artifacts = {
        "probability_map",
        "suspicious_mask",
        "overlay"
    }

    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != expected_artifacts
    ):
        raise DocTamperError(
            "Missing DocTamper visual artifacts."
        )

    total_bytes = 0

    for name in expected_artifacts:

        try:
            encoded = artifacts[name]

            if not isinstance(encoded, str):
                raise ValueError("Invalid artifact type.")

            decoded = base64.b64decode(
                encoded,
                validate=True
            )

        except (ValueError, binascii.Error) as exc:
            raise DocTamperError(
                "Invalid visual artifact."
            ) from exc

        if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DocTamperError(
                "Invalid PNG artifact."
            )

        total_bytes += len(decoded)

        if total_bytes > MAX_ARTIFACT_BYTES:
            raise DocTamperError(
                "Visual artifacts exceed size limit."
            )

    return result
