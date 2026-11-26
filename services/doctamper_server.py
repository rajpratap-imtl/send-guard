import base64
import hashlib
import io
import json
import math
import sys
import tempfile
import warnings

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from PIL import Image, ImageFile


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG = json.loads(
    (BASE_DIR / "config.json").read_text(
        encoding="utf-8-sig"
    )
)["doctamper"]

DOC_DIR = Path(CONFIG["directory"]).resolve()
CHECKPOINT = Path(CONFIG["checkpoint"]).resolve()

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_ARTIFACT_BYTES = 24 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


# Verify the trusted checkpoint before loading it.

MODEL_HASH = hashlib.sha256(
    CHECKPOINT.read_bytes()
).hexdigest()

if MODEL_HASH.lower() != CONFIG["model_sha256"].lower():
    raise RuntimeError("DocTamper checkpoint hash mismatch.")


# Import Srishant's original implementation.

sys.path.insert(0, str(DOC_DIR))

import run_doctamper_final as doctamper
from doctamper_service import analyze_document


# Load the model once at startup.

DEVICE = torch.device("cpu")

print("Loading DocTamper checkpoint...", flush=True)

MODEL = doctamper.load_model(
    str(CHECKPOINT),
    DEVICE
)

print("DocTamper model loaded successfully.", flush=True)
print("Parameters:", sum(p.numel() for p in MODEL.parameters()))
print("Device:", DEVICE)
print("SHA256:", MODEL_HASH)


def validate_image(data):
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            Image.DecompressionBombWarning
        )

        with Image.open(io.BytesIO(data)) as image:

            if image.format not in ("PNG", "JPEG"):
                raise ValueError("Unsupported image format.")

            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image dimensions exceed limit.")

            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multiple frames are unsupported.")

            image_format = image.format
            dimensions = image.size

            image.verify()

        with Image.open(io.BytesIO(data)) as image:
            image.load()

    return image_format, dimensions


def infer_document(data, image_format, dimensions):

    with tempfile.TemporaryDirectory(
        prefix="doctamper_"
    ) as temporary:

        directory = Path(temporary)

        suffix = ".png" if image_format == "PNG" else ".jpg"

        input_path = directory / ("document" + suffix)

        output_dir = directory / "outputs"

        input_path.write_bytes(data)

        # Use the original model and inference pipeline.
        # Suppress its diagnostic JSON containing local paths.

        with redirect_stdout(io.StringIO()):
            result = analyze_document(
                image_path=input_path,
                model=MODEL,
                device=DEVICE,
                output_dir=output_dir
            )

        if result.get("status") != "completed":
            raise RuntimeError("DocTamper inference failed.")

        signals = result["signals"]

        mean_score = signals["mean_pixel_probability"]
        flagged_area = signals["tampered_region_percentage"]

        if (
            type(mean_score) not in (int, float)
            or type(flagged_area) not in (int, float)
            or not math.isfinite(mean_score)
            or not math.isfinite(flagged_area)
            or not 0 <= mean_score <= 1
            or not 0 <= flagged_area <= 100
            or signals["threshold"] != 0.5
        ):
            raise RuntimeError("Invalid DocTamper signals.")

        # Return visual evidence without exposing local paths.

        artifacts = {}
        total_bytes = 0

        expected_files = {
            "probability_map": "document_probability.png",
            "suspicious_mask": "document_mask.png",
            "overlay": "document_overlay.png"
        }

        for key, filename in expected_files.items():

            artifact_path = output_dir / filename

            if not artifact_path.is_file():
                raise RuntimeError("Missing inference artifact.")

            total_bytes += artifact_path.stat().st_size

            if total_bytes > MAX_ARTIFACT_BYTES:
                raise RuntimeError("Artifacts exceed response limit.")

            with Image.open(artifact_path) as artifact:
                if (
                    artifact.format != "PNG"
                    or artifact.size != dimensions
                ):
                    raise RuntimeError("Invalid inference artifact.")

                artifact.verify()

            artifacts[key] = base64.b64encode(
                artifact_path.read_bytes()
            ).decode("ascii")

        return {
            "module": "document_tampering_detection",
            "model": result["model"],
            "status": "completed",
            "signals": signals,
            "requires_calibration": True,
            "document_fraud_confirmed": None,
            "automatic_block": False,
            "artifact_encoding": "base64",
            "artifact_content_type": "image/png",
            "artifacts": artifacts,
            "note": (
                "Experimental pixel-level tampering evidence. "
                "Document authenticity remains undetermined."
            )
        }


class DetectorHandler(BaseHTTPRequestHandler):

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def send_json(self, status, payload):

        body = json.dumps(
            payload,
            allow_nan=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type", "application/json"
        )
        self.send_header(
            "Content-Length", str(len(body))
        )
        self.send_header(
            "Cache-Control", "no-store"
        )
        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        if self.path != "/health":
            self.send_json(404, {"detail": "Not found."})
            return

        self.send_json(200, {
            "status": "ok",
            "model": "SmallUNet",
            "device": str(DEVICE),
            "model_sha256": MODEL_HASH,
            "threshold": 0.5
        })

    def do_POST(self):

        if self.path != "/predict":
            self.send_json(404, {"detail": "Not found."})
            return

        if (
            self.headers.get_content_type()
            != "application/octet-stream"
        ):
            self.send_json(415, {
                "detail": "Raw image bytes required."
            })
            return

        if self.headers.get("Transfer-Encoding"):
            self.send_json(400, {
                "detail": "Unsupported transfer encoding."
            })
            return

        try:
            size = int(
                self.headers.get("Content-Length", "-1")
            )
        except ValueError:
            size = -1

        if size < 0:
            self.send_json(400, {
                "detail": "Invalid content length."
            })
            return

        if size == 0:
            self.send_json(400, {
                "detail": "Empty image."
            })
            return

        if size > MAX_BYTES:
            self.send_json(413, {
                "detail": "Maximum image size is 10 MiB."
            })
            return

        try:
            data = self.rfile.read(size)
        except OSError:
            self.send_json(408, {
                "detail": "Image upload timed out."
            })
            return

        if len(data) != size:
            self.send_json(400, {
                "detail": "Incomplete image upload."
            })
            return

        try:
            image_format, dimensions = validate_image(data)

        except Exception:
            self.send_json(422, {
                "detail": "Unsupported, damaged or oversized image."
            })
            return

        try:
            result = infer_document(
                data,
                image_format,
                dimensions
            )

        except Exception as exc:
            print(
                "DocTamper inference error:",
                type(exc).__name__,
                flush=True
            )
            self.send_json(500, {
                "detail": "Document analysis failed."
            })
            return

        self.send_json(200, result)


if __name__ == "__main__":

    server = HTTPServer(
        ("127.0.0.1", 8002),
        DetectorHandler
    )

    print("========================================")
    print("DocTamper - SmallUNet")
    print("Mode: CPU")
    print("Address: http://127.0.0.1:8002")
    print("========================================", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping DocTamper service.")
    finally:
        server.server_close()
