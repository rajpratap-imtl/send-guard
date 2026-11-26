import json

from pathlib import Path
from threading import BoundedSemaphore

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Response
)

from adapters.doctamper_detector import (
    scan_document,
    check_doctamper_health,
    DocTamperError,
    MAX_DOCUMENT_BYTES
)


BASE_DIR = Path(__file__).resolve().parent

CONFIG = json.loads(
    (BASE_DIR / "config.json").read_text(
        encoding="utf-8-sig"
    )
)

router = APIRouter()

DOCUMENT_SLOT = BoundedSemaphore(1)


@router.get("/api/health/doctamper")
def doctamper_health():

    try:
        return check_doctamper_health(
            CONFIG["doctamper"]
        )

    except DocTamperError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc)
        ) from exc


@router.post("/api/scan/document")
def analyze_document(
    response: Response,
    file: UploadFile = File(...)
):

    if not DOCUMENT_SLOT.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="Document detector is busy."
        )

    try:
        data = file.file.read(
            MAX_DOCUMENT_BYTES + 1
        )

        try:
            result = scan_document(
                data,
                CONFIG["doctamper"]
            )

        except DocTamperError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc)
            ) from exc

        result["file_name"] = Path(
            file.filename or "uploaded_document"
        ).name

        response.headers["Cache-Control"] = "no-store"

        return result

    finally:
        DOCUMENT_SLOT.release()
