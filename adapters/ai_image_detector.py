import json
import math

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


MAX_IMAGE_BYTES = 10 * 1024 * 1024

EXPECTED_THRESHOLD = 0.8947610855102539


class AIImageDetectorError(RuntimeError):
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

    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(65537)

        if len(raw) > 65536:
            raise AIImageDetectorError(
                "Detector response exceeds expected size."
            )

        result = json.loads(raw)

    except HTTPError as exc:
        messages = {
            400: "Empty or invalid image.",
            413: "Image exceeds the 10 MiB limit.",
            422: "Unsupported, damaged or oversized image."
        }

        if exc.code in messages:
            raise AIImageDetectorError(
                messages[exc.code],
                exc.code
            ) from exc

        raise AIImageDetectorError(
            "AI image detector returned an error."
        ) from exc

    except (URLError, TimeoutError, OSError) as exc:
        raise AIImageDetectorError(
            "AI image detector is unavailable."
        ) from exc

    except ValueError as exc:
        raise AIImageDetectorError(
            "Invalid AI detector response."
        ) from exc

    if not isinstance(result, dict):
        raise AIImageDetectorError(
            "Unexpected AI detector response."
        )

    return result


def check_ai_health(config):
    result = request_detector(config, "/health")

    if result.get("status") != "ok":
        raise AIImageDetectorError(
            "AI image detector is unhealthy."
        )

    actual_hash = str(
        result.get("model_sha256", "")
    ).lower()

    expected_hash = config["model_sha256"].lower()

    if actual_hash != expected_hash:
        raise AIImageDetectorError(
            "AI image model integrity mismatch."
        )

    return result


def scan_ai_image(data, config):
    if not data:
        raise AIImageDetectorError(
            "Empty image.", 400
        )

    if len(data) > MAX_IMAGE_BYTES:
        raise AIImageDetectorError(
            "Image exceeds the 10 MiB limit.", 413
        )

    check_ai_health(config)

    result = request_detector(
        config,
        "/predict",
        data
    )

    score = result.get("ai_score")

    if (
        result.get("prediction")
        not in ("real", "ai_generated")
        or type(score) not in (float, int)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or result.get("threshold") != EXPECTED_THRESHOLD
        or result.get("is_calibrated_probability") is not False
        or result.get("automatic_block") is not False
    ):
        raise AIImageDetectorError(
            "Unexpected AI image prediction or policy."
        )

    expected_prediction = (
        "ai_generated"
        if score >= EXPECTED_THRESHOLD
        else "real"
    )

    if result["prediction"] != expected_prediction:
        raise AIImageDetectorError(
            "AI detector classification mismatch."
        )

    result["module"] = "ai_image_detection"
    result["model"] = "EfficientNetB0"
    result["status"] = "success"

    return result
