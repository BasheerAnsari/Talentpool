from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.recruiter_schema import RecruiterSignupSchema, RecruiterLoginSchema
from app.core.database import get_db
from app.services.recruiter_services import recruiter_signup_service, recruiter_signin_service
from app.shared.helper.error_logger import log_errors


router = APIRouter(prefix="/api", tags=["auth"])


@router.post("/signup", status_code=status.HTTP_201_CREATED)
@log_errors(stage="recruiter_signup_route")
def recruiter_signup(
    data: RecruiterSignupSchema,
    db: Session = Depends(get_db)
):
    try:
        recruiter = recruiter_signup_service(db, data)

        return {
            "message": "Recruiter created successfully",
            "recruiter_id": recruiter.id
        }

    except ValueError as ve:
        raise HTTPException(400, detail=str(ve))

    except Exception:
        raise HTTPException(500, detail="Internal server error")


@router.post("/signin")
@log_errors(stage="recruiter_signin_route")
def recruiter_signin(data: RecruiterLoginSchema, db: Session = Depends(get_db)):
    return recruiter_signin_service(data, db)