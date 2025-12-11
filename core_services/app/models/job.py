# models/job.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base  # shared Base
from app.models.recruiter_model import Recruiter

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True) 
    job_title = Column(String)
    department = Column(String)
    job_description = Column(String)
    required_skills = Column(String)  # store as CSV string for now
    experience_level = Column(String)
    location = Column(String)
    employment_type = Column(String)
    user_id = Column(Integer, ForeignKey("recruiters.id", ondelete= "CASCADE"), nullable=False)  # forignkey refereing the recruter table 

    user= relationship("Recruiter",backref="jobs")