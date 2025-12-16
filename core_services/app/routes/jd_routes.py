from fastapi import APIRouter, UploadFile, File
from app.services.jd_service import extract_jd

router = APIRouter(prefix="/api/jd", tags=["Job Description Extractor"])


@router.post("/extract")
async def extract_jd_route(file: UploadFile = File(...)):
    """
    Upload a JD (PDF/DOCX/TXT) → Extract structured fields + 8–10 line JD.
    """
    result = await extract_jd(file)
    return result