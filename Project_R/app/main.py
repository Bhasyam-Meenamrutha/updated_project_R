import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.mock_hubble import router as mock_hubble_router
from app.employee_service import enrich_employee, save_employee
from app.database import employee_directory_collection, employees_collection


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


app = FastAPI(
    title="Hubble Employee API",
    description="Mock Hubble Employee Integration with MongoDB",
    version="1.1.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Allows the React dashboard running on localhost:5173
# to communicate with the Hubble Mock API running on port 8001.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Hubble Mock Router
# ---------------------------------------------------------
app.include_router(mock_hubble_router)


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------
@app.get("/")
def home():
    return {
        "success": True,
        "message": "Hubble Employee API is running",
        "service": "mock-hubble",
    }


# ---------------------------------------------------------
# Load Employee Directory
# ---------------------------------------------------------
@app.post("/api/directory/load")
def load_directory():
    path = os.path.join(
        BASE_DIR,
        "data",
        "employee_directory.json",
    )

    try:
        with open(path, "r", encoding="utf-8") as file:
            people = json.load(file)

        for person in people:
            employee_directory_collection.update_one(
                {"name": person["name"]},
                {"$set": person},
                upsert=True,
            )

        return {
            "success": True,
            "message": "Employee directory loaded successfully",
            "count": len(people),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="employee_directory.json not found",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# Sync Employee
# ---------------------------------------------------------
@app.post("/api/employees/sync/{login_id}")
def sync_employee(login_id: str):
    path = os.path.join(
        BASE_DIR,
        "data",
        "mock_employees.json",
    )

    try:
        with open(path, "r", encoding="utf-8") as file:
            employees = json.load(file)

        hubble_employee = next(
            (
                employee
                for employee in employees
                if employee.get("loginId", "").lower()
                == login_id.lower()
            ),
            None,
        )

        if not hubble_employee:
            raise HTTPException(
                status_code=404,
                detail=f"Employee '{login_id}' not found in Hubble",
            )

        employee = enrich_employee(hubble_employee)
        saved_employee = save_employee(employee)

        return {
            "success": True,
            "message": "Employee synchronized successfully",
            "data": saved_employee,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# Get Employee by Hubble Login ID
# ---------------------------------------------------------
@app.get("/api/employees/{login_id}")
def get_employee(login_id: str):
    employee = employees_collection.find_one(
        {"loginId": login_id},
        {"_id": 0},
    )

    if not employee:
        raise HTTPException(
            status_code=404,
            detail=f"Employee '{login_id}' not found in MongoDB",
        )

    return {
        "success": True,
        "data": employee,
    }