from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.job import Job
from app.schemas.job import JOB
from app.shared.helper.pagination import Paginator


class JobService:

    @staticmethod
    def create_job(job: JOB, db: Session):
        try:
            db_job = Job(
                job_title=job.job_title,
                department=job.department,
                job_description=job.job_description,
                required_skills=job.required_skills,
                experience_level=job.experience_level,
                location=job.location,
                employment_type=job.employment_type,
                user_id=job.user_id
            )

            db.add(db_job)
            db.commit()
            db.refresh(db_job)

            return db_job

        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


    @staticmethod
    def get_jobs(user_id: int, job_title: str, location: str, page: int, per_page: int, db: Session):
        try:
            query = db.query(Job)

            filters = [Job.user_id == user_id]  

            if job_title:   
                filters.append(Job.job_title.ilike(f"%{job_title}%"))

            if location:    
                filters.append(Job.location.ilike(f"%{location}%"))
             
            query = query.filter(and_(*filters))

            if job_title:
                query = query.filter(Job.job_title.ilike(f"%{job_title}%"))

            if location:
                query = query.filter(Job.location.ilike(f"%{location}%"))

            return Paginator.pagination(query,page,per_page)


        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching jobs: {str(e)}")