from sqlalchemy.orm import Session
from passlib.hash import bcrypt
from app.models.recruiter_model import Recruiter
from app.shared.helper.error_logger import log_errors

from fastapi import HTTPException, status
from app.models.recruiter_model import Recruiter
from app.schemas.recruiter_schema import RecruiterLoginSchema
from app.shared.helper.error_logger import log_errors

@log_errors(stage="recruiter_signup_service")
def recruiter_signup_service(db: Session, data):

    if db.query(Recruiter).filter(
        Recruiter.username == data.credentials.username
    ).first():
        raise ValueError("Username already exists")

    raw_password = data.credentials.password
    safe_password = raw_password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    hashed_password = bcrypt.hash(safe_password)

    new_recruiter = Recruiter(
        company_name=data.companyInfo.companyName,
        company_id=data.companyInfo.companyId,
        company_address=data.companyInfo.companyAddress,
        contact_name=data.companyInfo.contact_name,
        contact_number=data.companyInfo.contactNumber,
        company_email=data.companyInfo.email,
        username=data.credentials.username,
        password=hashed_password,
        registration_cert_path=None,
        gst_cert_path=None
    )

    db.add(new_recruiter)
    db.commit()
    db.refresh(new_recruiter)

    return new_recruiter


@log_errors(stage="recruiter_signin_service")
def recruiter_signin_service(data: RecruiterLoginSchema, db: Session):
    # Find user by username
    user = db.query(Recruiter).filter(Recruiter.username == data.username).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # BCRYPT SAFE CHECK (72 BYTES LIMIT)
    raw_password = data.password
    safe_password = raw_password.encode("utf-8")[:72].decode("utf-8", errors="ignore")

    if not bcrypt.verify(safe_password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    return {
        "message": "Login successful",
        "user_id": user.id
    }