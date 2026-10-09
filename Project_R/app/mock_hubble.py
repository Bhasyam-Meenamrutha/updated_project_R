import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "mock_employees.json"

router = APIRouter(prefix="/mock-hubble", tags=["Mock Hubble"])


def load_employees():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@router.get("/employee/{login_id}")
def get_employee(login_id: str):
    for employee in load_employees():
        if employee.get("loginId", "").lower() == login_id.lower():
            return employee

    raise HTTPException(status_code=404, detail="Employee not found")
