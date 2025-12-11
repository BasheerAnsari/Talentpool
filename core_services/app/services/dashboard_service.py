from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.job import Job

def get_recruiter_dashboard_service():
    summary_cards = {
        "totalJobs": 220,
        "interviewsScheduled": 83,
        "projectsCompleted": 156,
        "offersAccepted": 94,
        "totalClosing": 128
    }

    donut_chart = {
        "title": "Applications of department",
        "labels": ["Engineering", "Sales", "Marketing", "Design", "Operations", "BPO"],
        "values": [25, 15, 20, 18, 12, 10]
    }

    line_chart = {
        "title": "Candidates Submission Funnel",
        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "values": {
            "LinkedIn": [20, 25, 40, 50, 60, 75, 80, 90, 100, 95, 110, 120],
            "Apna":     [15, 20, 30, 35, 45, 60, 70, 80, 85, 90, 100, 110],
            "Naukri":   [18, 22, 38, 45, 55, 65, 75, 89, 98, 105, 115, 125]
        }
    }

    jobs_table = [
        {
            "jobCode": "Job-001",
            "position": "UI/UX Designer",
            "department": "Design",
            "applicants": 124,
            "status": "In Progress",
            "lastUpdated": "2 hours ago"
        },
        {
            "jobCode": "Job-002",
            "position": "Backend Developer",
            "department": "Developer",
            "applicants": 89,
            "status": "Active",
            "lastUpdated": "Few minutes ago"
        },
        {
            "jobCode": "Job-003",
            "position": "Frontend Designer",
            "department": "Design",
            "applicants": 156,
            "status": "Active",
            "lastUpdated": "3 hours ago"
        }
    ]

    upcoming_interviews = [
        {
            "candidateName": "Saraha Williams",
            "position": "Frontend Developer",
            "status": "Pending",
            "date": "2025-01-12 14:00"
        },
        {
            "candidateName": "Saraha Williams",
            "position": "Frontend Developer",
            "status": "Pending",
            "date": "2025-01-13 12:00"
        },
        {
            "candidateName": "Saraha Williams",
            "position": "Backend Developer",
            "status": "Confirmed",
            "date": "2025-01-14 10:00"
        }
    ]

    return {
        "summaryCards": summary_cards,
        "donutChart": donut_chart,
        "lineChart": line_chart,
        "jobsTable": jobs_table,
        "upcomingInterviews": upcoming_interviews
    }
