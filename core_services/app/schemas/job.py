from pydantic import BaseModel, Field, conint
from typing import List

class JOB(BaseModel):
    job_title: str = Field(..., min_length=1, description="Job title cannot be empty")
    department: str = Field(..., min_length=1, description="Department cannot be empty")
    job_description: str = Field(..., min_length=1, description="Job description cannot be empty")
    required_skills: List[str] = Field(..., description="At least one skill is required")
    experience_level: str = Field(..., min_length=1, description="Experience level cannot be empty")
    location: str = Field(..., min_length=1, description="Location cannot be empty")
    employment_type: str = Field(..., min_length=1, description="Employment type cannot be empty")
    user_id: conint(gt=0) = Field(..., description="User ID must be greater than 0")

