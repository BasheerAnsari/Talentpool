from fastapi import APIRouter
from app.services.dashboard_service import get_recruiter_dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/recruiter")
def get_dashboard():
    data = get_recruiter_dashboard_service()
    return {
        "status": "success",
        "data": data
    }
