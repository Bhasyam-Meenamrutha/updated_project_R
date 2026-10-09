import os
import pickle
import shutil
import sqlite3

import cv2
import httpx
import numpy as np

from datetime import datetime

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

from fastapi.concurrency import run_in_threadpool

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from insightface.app import FaceAnalysis
from insightface.app.common import Face

from attendance import (
    get_connection,
    add_employee,
    get_employee,
    delete_attendance_by_date,
    record_entry,
    record_exit,
    get_today_attendance,
    get_employee_events,
    get_daily_summary,
    get_attendance_by_date
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    BASE_DIR,
    "attendance.db"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "embeddings",
    "face_embeddings.pkl"
)

FACES_DIRECTORY = os.path.join(
    BASE_DIR,
    "data",
    "faces_v2"
)

HUBBLE_API_URL = os.getenv(
    "HUBBLE_API_URL",
    "http://127.0.0.1:8001"
).rstrip("/")


# ============================================================
# TWO CAMERA CONFIGURATION
# ============================================================

ENTRY_CAMERA = "/dev/video0"

EXIT_CAMERA = "/dev/video2"


# ============================================================
# INSIGHTFACE
# ============================================================

print("Loading InsightFace model...")

face_app = FaceAnalysis(
    name="buffalo_l"
)

face_app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("InsightFace model loaded.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Employee Attendance API",
    description=(
        "Backend API for the Employee Attendance System"
    ),
    version="6.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class EmployeeCreate(BaseModel):

    employee_id: str
    employee_name: str = ""
    shift_start: str
    shift_end: str = "22:00"
    email: str = ""
    login_id: str = ""


class AttendanceEventCreate(BaseModel):

    employee_id: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Employee Attendance API is running",

        "version":
            "6.0.0",

        "architecture":
            "two-camera"
    }


# ============================================================
# CAMERA STATUS
# ============================================================

@app.get("/cameras/status")
def cameras_status():

    entry_exists = os.path.exists(
        ENTRY_CAMERA
    )

    exit_exists = os.path.exists(
        EXIT_CAMERA
    )

    return {

        "architecture":
            "two-camera",

        "entry_camera": {

            "device":
                ENTRY_CAMERA,

            "mode":
                "ENTRY",

            "responsibility":
                "CHECK_IN",

            "device_exists":
                entry_exists
        },

        "exit_camera": {

            "device":
                EXIT_CAMERA,

            "mode":
                "EXIT",

            "responsibility":
                "CHECK_OUT",

            "device_exists":
                exit_exists
        }
    }


# ============================================================
# GET ALL EMPLOYEES
# ============================================================

@app.get("/employees")
def get_employees():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            employee_id,
            employee_name,
            shift_start,
            shift_end,
            email,
            login_id,
            hubble_id,
            reports_to,
            reports_to_mail_id,
            hr_contact,
            hr_email,
            department,
            designation,
            team_name
        FROM employees
        ORDER BY employee_id
    """)

    employees = cursor.fetchall()

    connection.close()

    return [
        dict(employee)
        for employee in employees
    ]


# ============================================================
# HUBBLE INTEGRATION
# ============================================================

def sync_hubble_employee(login_id: str):

    try:
        response = httpx.post(
            f"{HUBBLE_API_URL}/api/employees/sync/{login_id}",
            timeout=20.0
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to connect to the Hubble employee API. "
                f"Start it on port 8001 or set HUBBLE_API_URL. ({exc})"
            )
        )

    if not response.is_success:
        try:
            detail = response.json().get(
                "detail",
                "Employee synchronization failed"
            )
        except ValueError:
            detail = "Employee synchronization failed"

        raise HTTPException(
            status_code=502,
            detail=str(detail)
        )

    payload = response.json()
    return payload.get("data", {})


@app.get("/hubble/status")
def hubble_status():
    try:
        response = httpx.get(
            f"{HUBBLE_API_URL}/",
            timeout=5.0
        )
        return {
            "configured": True,
            "url": HUBBLE_API_URL,
            "available": response.is_success,
            "response": response.json() if response.is_success else None
        }
    except Exception as exc:
        return {
            "configured": True,
            "url": HUBBLE_API_URL,
            "available": False,
            "error": str(exc)
        }


# ============================================================
# CREATE EMPLOYEE
# ============================================================

@app.post("/employees")
def create_employee(
    employee: EmployeeCreate
):

    try:
        datetime.strptime(employee.shift_start, "%H:%M")
        datetime.strptime(employee.shift_end, "%H:%M")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="shift_start and shift_end must use HH:MM format, e.g. 14:00"
        )

    employee_id = employee.employee_id.strip()
    login_id = employee.login_id.strip()

    if not employee_id:
        raise HTTPException(
            status_code=400,
            detail="employee_id is required"
        )

    existing_employee = get_employee(employee_id)

    if existing_employee:
        raise HTTPException(
            status_code=409,
            detail="Employee already exists"
        )

    hubble_data = {}

    # Hubble is an additive integration. Existing Project_R
    # manual employee creation continues to work when login_id
    # is empty. Supplying a login_id activates Hubble sync.
    if login_id:
        hubble_data = sync_hubble_employee(login_id)

    employee_name = (
        hubble_data.get("name")
        or employee.employee_name.strip()
    )
    email = (
        hubble_data.get("mailId")
        or employee.email.strip()
    )

    if not employee_name:
        raise HTTPException(
            status_code=400,
            detail="employee_name is required when Hubble sync is not used"
        )

    add_employee(
        employee_id=employee_id,
        employee_name=employee_name,
        shift_start=employee.shift_start,
        shift_end=employee.shift_end,
        email=email,
        login_id=(hubble_data.get("loginId") or login_id or None),
        hubble_id=str(hubble_data.get("id")) if hubble_data.get("id") is not None else None,
        reports_to=hubble_data.get("reportsTo"),
        reports_to_mail_id=hubble_data.get("reportsToMailId"),
        hr_contact=hubble_data.get("hrContact"),
        hr_email=hubble_data.get("hrEmail"),
        department=hubble_data.get("department"),
        designation=hubble_data.get("designation"),
        team_name=hubble_data.get("teamName")
    )

    return {
        "message": "Employee created successfully",
        "employee": {
            "employee_id": employee_id,
            "login_id": hubble_data.get("loginId") or login_id or None,
            "employee_name": employee_name,
            "shift_start": employee.shift_start,
            "shift_end": employee.shift_end,
            "email": email,
            "hubble": hubble_data or None
        }
    }


# ============================================================
# DELETE EMPLOYEE COMPLETELY
# ============================================================

@app.delete(
    "/employees/{employee_id}"
)
def remove_employee(
    employee_id: str
):

    """
    Completely delete an employee.

    Deletes:

        - Employee record
        - Attendance events
        - Daily attendance
        - Face embedding
        - Face directory
    """

    # --------------------------------------------------------
    # OPEN THE ABSOLUTE DATABASE DIRECTLY
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    # --------------------------------------------------------
    # CHECK EMPLOYEE
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            employee_id,
            employee_name,
            shift_start,
            shift_end,
            email
        FROM employees
        WHERE TRIM(employee_id) = TRIM(?)
        """,
        (
            employee_id,
        )
    )

    employee = cursor.fetchone()

    # --------------------------------------------------------
    # EMPLOYEE NOT FOUND
    # --------------------------------------------------------

    if employee is None:

        cursor.execute("""
            SELECT
                employee_id,
                employee_name
            FROM employees
            ORDER BY employee_id
        """)

        existing_employees = cursor.fetchall()

        connection.close()

        raise HTTPException(
            status_code=404,
            detail={

                "message":
                    (
                        f"Employee {employee_id} "
                        "was not found."
                    ),

                "database_used":
                    DATABASE,

                "employees_in_database": [

                    {
                        "employee_id":
                            row["employee_id"],

                        "employee_name":
                            row["employee_name"]
                    }

                    for row in existing_employees
                ]
            }
        )

    # --------------------------------------------------------
    # DELETE ATTENDANCE EVENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM attendance_events
        WHERE TRIM(employee_id) = TRIM(?)
        """,
        (
            employee_id,
        )
    )

    events_deleted = cursor.rowcount

    # --------------------------------------------------------
    # DELETE DAILY ATTENDANCE
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM daily_attendance
        WHERE TRIM(employee_id) = TRIM(?)
        """,
        (
            employee_id,
        )
    )

    summaries_deleted = cursor.rowcount

    # --------------------------------------------------------
    # DELETE EMPLOYEE
    # --------------------------------------------------------

    cursor.execute(
        """
        DELETE FROM employees
        WHERE TRIM(employee_id) = TRIM(?)
        """,
        (
            employee_id,
        )
    )

    employee_deleted = cursor.rowcount

    # --------------------------------------------------------
    # COMMIT DATABASE CHANGES
    # --------------------------------------------------------

    connection.commit()

    connection.close()

    # --------------------------------------------------------
    # DELETE FACE EMBEDDING
    # --------------------------------------------------------

    embedding_deleted = False

    if os.path.exists(
        EMBEDDINGS_FILE
    ):

        try:

            with open(
                EMBEDDINGS_FILE,
                "rb"
            ) as file_object:

                employee_embeddings = pickle.load(
                    file_object
                )

            if isinstance(
                employee_embeddings,
                dict
            ):

                matching_key = None

                # Exact match first
                if employee_id in employee_embeddings:

                    matching_key = employee_id

                else:

                    # Check for accidental whitespace
                    for key in employee_embeddings:

                        if (
                            str(key).strip()
                            == employee_id.strip()
                        ):

                            matching_key = key
                            break

                if matching_key is not None:

                    del employee_embeddings[
                        matching_key
                    ]

                    embedding_deleted = True

                    with open(
                        EMBEDDINGS_FILE,
                        "wb"
                    ) as file_object:

                        pickle.dump(
                            employee_embeddings,
                            file_object
                        )

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Employee was deleted from the "
                    "database, but the face embedding "
                    "could not be updated: "
                    f"{error}"
                )
            )

    # --------------------------------------------------------
    # DELETE FACE DIRECTORY
    # --------------------------------------------------------

    face_directory = os.path.join(
        FACES_DIRECTORY,
        employee_id
    )

    face_directory_deleted = False

    if os.path.exists(
        face_directory
    ):

        try:

            shutil.rmtree(
                face_directory
            )

            face_directory_deleted = True

        except Exception as error:

            print(
                "Face directory deletion failed:",
                error
            )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "message":
            "Employee deleted completely",

        "employee_id":
            employee_id,

        "employee_name":
            employee["employee_name"],

        "database_used":
            DATABASE,

        "deleted": {

            "employee_record":
                employee_deleted,

            "attendance_events":
                events_deleted,

            "daily_attendance":
                summaries_deleted,

            "face_embedding":
                embedding_deleted,

            "face_directory":
                face_directory_deleted
        }
    }


# ============================================================
# RECORD ENTRY
# ============================================================

@app.post(
    "/attendance/entry"
)
def attendance_entry(
    attendance: AttendanceEventCreate
):

    employee = get_employee(
        attendance.employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    success = record_entry(
        attendance.employee_id
    )

    if not success:

        return {

            "message":
                "Entry was not recorded",

            "employee_id":
                attendance.employee_id,

            "status":
                "IGNORED"
        }

    summary = get_daily_summary(
        attendance.employee_id
    )

    return {

        "message":
            "Entry recorded successfully",

        "employee_id":
            attendance.employee_id,

        "event_type":
            "IN",

        "camera":
            ENTRY_CAMERA,

        "summary":
            summary
    }


# ============================================================
# RECORD EXIT
# ============================================================

@app.post(
    "/attendance/exit"
)
def attendance_exit(
    attendance: AttendanceEventCreate
):

    employee = get_employee(
        attendance.employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    success = record_exit(
        attendance.employee_id
    )

    if not success:

        return {

            "message":
                "Exit was not recorded",

            "employee_id":
                attendance.employee_id,

            "status":
                "IGNORED"
        }

    summary = get_daily_summary(
        attendance.employee_id
    )

    return {

        "message":
            "Exit recorded successfully",

        "employee_id":
            attendance.employee_id,

        "event_type":
            "OUT",

        "camera":
            EXIT_CAMERA,

        "summary":
            summary
    }


# ============================================================
# TODAY ATTENDANCE
# ============================================================

@app.get(
    "/attendance/today"
)
def today_attendance():

    return get_today_attendance()


# ============================================================
# HISTORICAL ATTENDANCE
# ============================================================

@app.get(
    "/attendance/date/{attendance_date}"
)
def attendance_by_date(
    attendance_date: str
):

    # --------------------------------------------------------
    # VALIDATE DATE
    # --------------------------------------------------------

    try:

        datetime.strptime(
            attendance_date,
            "%Y-%m-%d"
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid date format. "
                "Use YYYY-MM-DD."
            )
        )

    # --------------------------------------------------------
    # GET ATTENDANCE
    # --------------------------------------------------------

    try:

        result = get_attendance_by_date(
            attendance_date
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    return result


# ============================================================
# DELETE ATTENDANCE FOR A PARTICULAR DATE
# ============================================================

@app.delete(
    "/attendance/date/{attendance_date}"
)
def remove_attendance_date(
    attendance_date: str
):

    """
    Delete attendance for a specific date.

    Deletes:
        - attendance_events
        - daily_attendance

    Keeps:
        - employees
        - face embeddings
        - employee face enrollment
    """

    # --------------------------------------------------------
    # VALIDATE DATE
    # --------------------------------------------------------

    try:

        datetime.strptime(
            attendance_date,
            "%Y-%m-%d"
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid date format. "
                "Use YYYY-MM-DD."
            )
        )

    # --------------------------------------------------------
    # DELETE ATTENDANCE
    # --------------------------------------------------------

    try:

        result = delete_attendance_by_date(
            attendance_date
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "message":
            "Attendance deleted successfully",

        "date":
            attendance_date,

        "events_deleted":
            result["events_deleted"],

        "summaries_deleted":
            result["summaries_deleted"]
    }


# ============================================================
# EMPLOYEE DAILY SUMMARY
# ============================================================

@app.get(
    "/attendance/{employee_id}/summary"
)
def employee_daily_summary(
    employee_id: str
):

    employee = get_employee(
        employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    summary = get_daily_summary(
        employee_id
    )

    return summary


# ============================================================
# EMPLOYEE EVENTS
# ============================================================

@app.get(
    "/attendance/{employee_id}/events"
)
def employee_events(
    employee_id: str,
    attendance_date: str | None = None
):

    employee = get_employee(
        employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    # --------------------------------------------------------
    # VALIDATE DATE
    # --------------------------------------------------------

    if attendance_date is not None:

        try:

            datetime.strptime(
                attendance_date,
                "%Y-%m-%d"
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "attendance_date must use "
                    "YYYY-MM-DD format"
                )
            )

    events = get_employee_events(
        employee_id,
        attendance_date
    )

    return [
        dict(event)
        for event in events
    ]


# ============================================================
# EMPLOYEE ATTENDANCE
# ============================================================

@app.get(
    "/attendance/{employee_id}"
)
def employee_attendance(
    employee_id: str
):

    employee = get_employee(
        employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    events = get_employee_events(
        employee_id
    )

    return [
        dict(event)
        for event in events
    ]


# ============================================================
# ALL ATTENDANCE EVENTS
# ============================================================

@app.get(
    "/attendance"
)
def all_attendance():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            attendance_events.id,
            attendance_events.employee_id,
            employees.employee_name,
            employees.shift_start,
            employees.shift_end,
            attendance_events.event_date,
            attendance_events.event_time,
            attendance_events.event_type,
            attendance_events.created_at
        FROM attendance_events
        LEFT JOIN employees
        ON attendance_events.employee_id =
           employees.employee_id
        ORDER BY
            attendance_events.event_date DESC,
            attendance_events.event_time DESC,
            attendance_events.id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return [
        dict(record)
        for record in records
    ]


# ============================================================
# LIVE HEAD-POSE CHECK (used by guided face enrollment)
# ============================================================

def detect_faces_with_pose(image, max_faces=5):

    """
    Fast face + head-pose detection for the live enrollment guide.

    face_app.get() also runs the recognition (embedding) and
    gender/age models on every face, which is slow and not needed
    here. This runs only:

        1. the face detector
        2. the 3D-68 landmark model (gives pose + landmarks)

    Falls back to face_app.get() if the internals are not
    available in the installed insightface version.
    """

    try:

        bboxes, kpss = face_app.det_model.detect(
            image,
            max_num=0,
            metric="default"
        )

        if bboxes is None or bboxes.shape[0] == 0:
            return []

        areas = (
            (bboxes[:, 2] - bboxes[:, 0]) *
            (bboxes[:, 3] - bboxes[:, 1])
        )

        order = np.argsort(-areas)[:max_faces]

        landmark_model = face_app.models.get("landmark_3d_68")

        faces = []

        for index in order:

            face = Face(
                bbox=bboxes[index, 0:4],
                kps=kpss[index] if kpss is not None else None,
                det_score=bboxes[index, 4]
            )

            if landmark_model is not None:
                landmark_model.get(image, face)

            faces.append(face)

        return faces

    except Exception as error:

        print("Fast pose path failed, using face_app.get:", error)

        return list(face_app.get(image))


def landmark_cues(face):

    """
    Direction cues from the 68 landmarks.

    These do NOT depend on InsightFace's pose sign convention:

        yaw_cue   > 0 : nose is towards the RIGHT of the image
                        (= the person turned to THEIR left)
        pitch_cue smaller : chin is UP, bigger : chin is DOWN

    Used only to decide the direction (sign). The angle size
    still comes from the InsightFace pose.
    """

    landmarks = getattr(face, "landmark_3d_68", None)

    if landmarks is None:
        return None, None

    points = np.asarray(landmarks)[:, :2]

    jaw_width = max(abs(points[16, 0] - points[0, 0]), 1.0)

    yaw_cue = (
        points[30, 0] -
        (points[0, 0] + points[16, 0]) / 2.0
    ) / jaw_width

    face_height = max(points[8, 1] - points[27, 1], 1.0)

    pitch_cue = (
        points[30, 1] - points[27, 1]
    ) / face_height

    return float(yaw_cue), float(pitch_cue)


@app.post("/face-pose")
async def face_pose(
    file: UploadFile = File(...)
):

    """
    Take ONE small camera frame and report EVERY face in it
    (largest first) with its box, head pose and direction cues.

    The dashboard decides which face is the employee (the one
    inside the on-screen circle) and ignores everybody else.

    Nothing is saved.
    """

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty image."
        )

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    height, width = image.shape[:2]

    faces = await run_in_threadpool(
        detect_faces_with_pose,
        image
    )

    results = []

    for face in faces:

        x1, y1, x2, y2 = [float(v) for v in face.bbox]

        item = {
            "bbox": {
                "x1": max(0.0, x1 / width),
                "y1": max(0.0, y1 / height),
                "x2": min(1.0, x2 / width),
                "y2": min(1.0, y2 / height),
            },
            "det_score": float(face.det_score),
            "pose": None,
            "yaw_cue": None,
            "pitch_cue": None,
        }

        pose = getattr(face, "pose", None)

        if pose is not None:

            # InsightFace order is [pitch, yaw, roll]
            item["pose"] = {
                "pitch": float(pose[0]),
                "yaw": float(pose[1]),
                "roll": float(pose[2]),
            }

            yaw_cue, pitch_cue = landmark_cues(face)

            item["yaw_cue"] = yaw_cue
            item["pitch_cue"] = pitch_cue

        results.append(item)

    return {
        "face_count": len(results),
        "faces": results,
    }


# ============================================================
# MULTI-PHOTO FACE ENROLLMENT
# ============================================================

@app.post(
    "/employees/{employee_id}/enroll-face"
)
async def enroll_face(
    employee_id: str,
    files: list[UploadFile] = File(...)
):

    """
    Enroll multiple face images.

    Maximum:
        10 images

    Each image must contain:
        exactly one face.

    The final embedding is the normalized
    average of all valid embeddings.
    """

    # --------------------------------------------------------
    # CHECK EMPLOYEE
    # --------------------------------------------------------

    employee = get_employee(
        employee_id
    )

    if employee is None:

        raise HTTPException(
            status_code=404,
            detail="Employee not found"
        )

    # --------------------------------------------------------
    # CHECK FILE COUNT
    # --------------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide at least "
                "one face image."
            )
        )

    if len(files) > 10:

        raise HTTPException(
            status_code=400,
            detail=(
                "Maximum 10 face images "
                "can be uploaded at once."
            )
        )

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    embeddings = []

    processed_files = []

    rejected_files = []

    for file in files:

        # ----------------------------------------------------
        # VALIDATE FILE TYPE
        # ----------------------------------------------------

        if not file.content_type:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "File type could not be determined."
            })

            continue

        if not file.content_type.startswith(
            "image/"
        ):

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "File is not an image."
            })

            continue

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "Empty image."
            })

            continue

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "Invalid image."
            })

            continue

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        faces = face_app.get(
            image
        )

        if len(faces) == 0:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "No face detected."
            })

            continue

        if len(faces) > 1:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason": (
                    "Multiple faces detected. "
                    "Use an image containing "
                    "only the employee."
                )
            })

            continue

        # ----------------------------------------------------
        # GET EMBEDDING
        # ----------------------------------------------------

        embedding = (
            faces[0].normed_embedding
        )

        if embedding is None:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason":
                    "Could not generate face embedding."
            })

            continue

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        # ----------------------------------------------------
        # VALIDATE EMBEDDING
        # ----------------------------------------------------

        if embedding.shape[0] != 512:

            rejected_files.append({

                "filename":
                    file.filename,

                "reason": (
                    "Unexpected embedding dimension: "
                    f"{embedding.shape[0]}"
                )
            })

            continue

        # ----------------------------------------------------
        # ACCEPT EMBEDDING
        # ----------------------------------------------------

        embeddings.append(
            embedding
        )

        processed_files.append(
            file.filename
        )

    # --------------------------------------------------------
    # REQUIRE VALID PHOTO
    # --------------------------------------------------------

    if len(embeddings) == 0:

        raise HTTPException(

            status_code=400,

            detail={

                "message":
                    "No valid face images were processed.",

                "processed_files":
                    [],

                "rejected_files":
                    rejected_files
            }
        )

    # --------------------------------------------------------
    # CONVERT TO ARRAY
    # --------------------------------------------------------

    embeddings_array = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # AVERAGE EMBEDDINGS
    # --------------------------------------------------------

    average_embedding = np.mean(
        embeddings_array,
        axis=0
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    norm = np.linalg.norm(
        average_embedding
    )

    if norm == 0:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated embedding "
                "has zero magnitude."
            )
        )

    average_embedding = (
        average_embedding / norm
    ).astype(np.float32)

    # --------------------------------------------------------
    # CREATE EMBEDDING DIRECTORY
    # --------------------------------------------------------

    embeddings_directory = os.path.dirname(
        EMBEDDINGS_FILE
    )

    os.makedirs(
        embeddings_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD EXISTING EMBEDDINGS
    # --------------------------------------------------------

    employee_embeddings = {}

    if os.path.exists(
        EMBEDDINGS_FILE
    ):

        try:

            with open(
                EMBEDDINGS_FILE,
                "rb"
            ) as file_object:

                loaded_embeddings = pickle.load(
                    file_object
                )

            if isinstance(
                loaded_embeddings,
                dict
            ):

                employee_embeddings = (
                    loaded_embeddings
                )

        except (
            pickle.PickleError,
            EOFError,
            ValueError,
            TypeError
        ):

            employee_embeddings = {}

    # --------------------------------------------------------
    # SAVE FINAL EMBEDDING
    # --------------------------------------------------------

    employee_embeddings[
        employee_id
    ] = average_embedding

    with open(
        EMBEDDINGS_FILE,
        "wb"
    ) as file_object:

        pickle.dump(
            employee_embeddings,
            file_object
        )

    # --------------------------------------------------------
    # CREATE FACE DIRECTORY
    # --------------------------------------------------------

    employee_face_directory = os.path.join(
        FACES_DIRECTORY,
        employee_id
    )

    os.makedirs(
        employee_face_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "message":
            "Face enrollment completed successfully",

        "employee_id":
            employee_id,

        "employee_name":
            employee["employee_name"],

        "photos_received":
            len(files),

        "photos_processed":
            len(processed_files),

        "photos_rejected":
            len(rejected_files),

        "embedding_dimension":
            int(
                average_embedding.shape[0]
            ),

        "embedding_file":
            EMBEDDINGS_FILE,

        "processed_files":
            processed_files,

        "rejected_files":
            rejected_files
    }