from app.database import (
    employees_collection,
    employee_directory_collection
)


def normalize_name(name: str) -> str:

    if not name:
        return ""

    return " ".join(
        name.split()
    ).strip().lower()


def find_person_by_name(name: str):

    normalized_name = normalize_name(name)

    people = employee_directory_collection.find(
        {},
        {
            "_id": 0
        }
    )

    for person in people:

        directory_name = normalize_name(
            person.get("name", "")
        )

        if directory_name == normalized_name:

            return person

    return None


def enrich_employee(hubble_data: dict):

    # Find Reporting Manager
    manager = find_person_by_name(
        hubble_data.get("reportsTo")
    )

    reports_to_mail_id = None

    if manager:

        reports_to_mail_id = manager.get(
            "mailId"
        )


    # Find HR
    hr = find_person_by_name(
        hubble_data.get("hrContact")
    )

    hr_email = None

    if hr:

        hr_email = hr.get(
            "mailId"
        )


    # Final employee object
    employee = {

        "id": hubble_data.get("id"),

        "name": hubble_data.get("name"),

        "loginId":
            hubble_data.get("loginId"),

        "mailId":
            hubble_data.get("mailId"),

        "reportsTo":
            hubble_data.get("reportsTo"),

        "reportsToMailId":
            reports_to_mail_id,

        "hrContact":
            hubble_data.get("hrContact"),

        "hrEmail":
            hr_email,

        "department":
            hubble_data.get("department"),

        "designation":
            hubble_data.get("designation"),

        "teamName":
            hubble_data.get("teamName")
    }

    return employee


def save_employee(employee: dict):

    employees_collection.update_one(

        {
            "id": employee["id"]
        },

        {
            "$set": employee
        },

        upsert=True
    )

    return employee