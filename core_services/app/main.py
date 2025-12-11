from fastapi import FastAPI
from app.routes.recruiter_routes import router as recruiter_router
from app.routes.dashboard_routes import router as dashboard_router
from app.core.database import Base, engine
from app.models.recruiter_model import Recruiter
from app.routes.job_router import job_router
from app.config import get_settings
from fastapi.middleware.cors import CORSMiddleware

settings = get_settings()

app = FastAPI(title="TalentPool365 Recruiter Service")

# ✅ Create DB Tables
Base.metadata.create_all(bind=engine)

# ✅ Register Routes
app.include_router(recruiter_router)

# Signin Routes
#app.include_router(recruiter_router)

# job addition / job routs 
app.include_router(job_router)

# recruiter dashbord 
app.include_router(dashboard_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains (for development)
    allow_credentials=True,
    allow_methods=["*"],  # allow GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],  # allow all headers
)

@app.get("/")
def root():
    return {
        "status": "Recruiter Service Running",
        "environment": settings.ENVIRONMENT
    }
