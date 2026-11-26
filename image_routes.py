import json

from pathlib import Path
from threading import BoundedSemaphore

from fastapi import APIRouter, File, UploadFile, HTTPException

from adapters.ai_image_detector import (
    scan_ai_image,
    check_ai_health,
    AIImageDetectorError,
    MAX_IMAGE_BYTES
)


BASE_DIR = Path(__file__).resolve().parent

CONFIG = json.loads(
    (BASE_DIR / "config.json").read_text(
        encoding="utf-8-sig"
    )
)

router = APIRouter()

IMAGE_SLOT = BoundedSemaphore(1)


@router.get("/api/health/ai-image")
def ai_health():

    try:
        return check_ai_health(
            CONFIG["ai_image"]
        )

    except AIImageDetectorError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc)
        ) from exc


@router.post("/api/scan/image")
def analyze_image(file: UploadFile = File(...)):

    if not IMAGE_SLOT.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Image detector is busy."
        )

    try:
        data = file.file.read(
            MAX_IMAGE_BYTES + 1
        )

        try:
            result = scan_ai_image(
                data,
                CONFIG["ai_image"]
            )

        except AIImageDetectorError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc)
            ) from exc

        result["file_name"] = Path(
            file.filename or "uploaded_image"
        ).name

        return result

    finally:
        IMAGE_SLOT.release()
