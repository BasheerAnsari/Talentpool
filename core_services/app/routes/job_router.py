from fastapi import APIRouter, Depends, HTTPException,Query
from sqlalchemy.orm import Session
from app.schemas.job import JOB
from app.models.job import Job  # SQLAlchemy model
from app.core.database import get_db  
from app.services.job_service import JobService


job_router = APIRouter(prefix="/api/job", tags=["jobs"])


# Create Job
@job_router.post("/")
async def create_job(job: JOB, db: Session = Depends(get_db)):
    created_job = JobService.create_job(job, db)
    return {"message": "Job created successfully", "job_id": created_job.id}


# Get Job List
@job_router.get("/")
async def get_jobs(
    job_title: str | None = Query(default=None, description="Filter by job title"),
    location: str | None = Query(default=None, description="Filter by job location"),
    user_id: int | None = Query(...,description= "user_id is required"),
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=10, ge=1, le=100, description="Number of records per page"),
    db: Session = Depends(get_db)
):
    return JobService.get_jobs(user_id,job_title, location, page, per_page, db)