import io
import warnings

from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile
)

from PIL import Image

from image_routes import analyze_image
from document_routes import analyze_document
from decision_engine import make_recommendation


MAX_PE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def identify_file(file):

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size == 0:
        raise HTTPException(400, "Empty file.")

    if size > MAX_PE_BYTES:
        raise HTTPException(
            413,
            "Maximum supported file size is 100 MiB."
        )

    header = file.file.read(16)
    file.file.seek(0)

    # The existing PE endpoint performs additional
    # PE header and signature validation.

    if header.startswith(b"MZ"):
        return "windows_pe", size

    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = header.startswith(b"\xff\xd8\xff")

    if not (is_png or is_jpeg):
        raise HTTPException(
            422,
            "Unsupported file. Use a Windows PE, PNG or JPEG file."
        )

    if size > MAX_IMAGE_BYTES:
        raise HTTPException(
            413,
            "Maximum image size is 10 MiB."
        )

    data = file.file.read(MAX_IMAGE_BYTES + 1)
    file.file.seek(0)

    try:
        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(
                io.BytesIO(data),
                formats=["PNG", "JPEG"]
            ) as image:

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image dimensions exceed limit.")

                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multiple frames unsupported.")

                image_format = image.format
                image.verify()

    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning
    ) as exc:

        raise HTTPException(
            422,
            "Unsupported, damaged or oversized image."
        ) from exc

    return image_format.lower(), size


def run_detector(function, reject_invalid=False):

    try:
        result = function()

        return {
            "status": "completed",
            "result": result
        }

    except HTTPException as exc:

        if reject_invalid and exc.status_code in (400, 413, 422):
            raise

        return {
            "status": "error",
            "error": {
                "status_code": exc.status_code,
                "message": str(exc.detail)
            }
        }


def create_unified_router(analyze_executable):

    router = APIRouter()

    @router.post("/api/scan")
    def scan_file(
        response: Response,
        file: UploadFile = File(...),
        document: bool = Form(False)
    ):

        file_type, size = identify_file(file)

        results = {
            "malware_detection": {
                "status": "not_applicable"
            },
            "ai_image_detection": {
                "status": "not_applicable"
            },
            "document_tampering_detection": {
                "status": "not_applicable"
            }
        }

        if file_type == "windows_pe":

            file.file.seek(0)

            results["malware_detection"] = run_detector(
                lambda: analyze_executable(file),
                reject_invalid=True
            )

        else:

            file.file.seek(0)

            results["ai_image_detection"] = run_detector(
                lambda: analyze_image(file)
            )

            if document:

                file.file.seek(0)

                results["document_tampering_detection"] = run_detector(
                    lambda: analyze_document(response, file)
                )

            else:

                results["document_tampering_detection"] = {
                    "status": "not_requested"
                }

        completed = sum(
            item["status"] == "completed"
            for item in results.values()
        )

        failed = sum(
            item["status"] == "error"
            for item in results.values()
        )

        if failed and completed:
            analysis_status = "partial"

        elif failed:
            analysis_status = "failed"

        else:
            analysis_status = "completed"

        recommendation = make_recommendation(
            results,
            analysis_status
        )

        response.headers["Cache-Control"] = "no-store"

        return {
            "file_name": Path(
                file.filename or "uploaded_file"
            ).name,
            "file_type": file_type,
            "file_size_bytes": size,
            "analysis_status": analysis_status,
            "detectors_completed": completed,
            "detectors_failed": failed,
            "results": results,
            "recommendation": recommendation,
            "automatic_block": False
        }

    return router
