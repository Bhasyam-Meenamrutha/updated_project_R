# Employee Face Recognition Attendance System — Merged Project

This is the merged version of **Project_R** with the additional **Hubble Mock Employee API / MongoDB integration** from the FaceDetection project.

## What is included

### Existing Project_R functionality
- InsightFace face recognition
- Liveness / anti-spoofing
- Two-camera ENTRY / EXIT architecture
- Event-based IN/OUT attendance
- Multiple sessions per day
- Early / ON_TIME / LATE calculation
- Worked-time calculation
- Email notifications
- Multi-photo face enrollment
- Guided head-pose enrollment
- FastAPI backend
- React HR dashboard
- Employee deletion
- Attendance history and date-wise attendance

### Added Hubble functionality
- Mock Hubble employee endpoint
- Hubble employee synchronization by login ID
- MongoDB employee persistence
- Employee-directory enrichment
- Reporting-manager email enrichment
- HR-contact email enrichment
- Department, designation and team information
- Hubble status endpoint
- Optional Hubble sync during employee creation

## Architecture

```text
React HR Dashboard
        |
        v
Project_R FastAPI :8000
        |
        +--> SQLite attendance.db
        |
        +--> InsightFace / enrollment / attendance
        |
        +--> Hubble API :8001 (when login_id is supplied)
                    |
                    +--> mock_hubble.py
                    +--> mock_employees.json
                    +--> MongoDB Atlas
```

## Run the project

### 1. Create the Python environment

```bash
cd FaceDetection_Merged
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you already have the Project_R virtual environment with InsightFace installed, it can be reused instead.

### 2. Configure environment variables

Create `.env` from `.env.example` and add your real email settings.

Create `.env.hubble` from `.env.hubble.example` and add the MongoDB connection string.

**Never commit `.env`, `.env.hubble`, API keys, passwords, or MongoDB credentials.**

### 3. Start the Hubble mock API

From the project root:

```bash
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

The Hubble API will be available at:

```text
http://127.0.0.1:8001
```

Useful endpoints:

```text
GET  /mock-hubble/employee/{login_id}
POST /api/directory/load
POST /api/employees/sync/{login_id}
GET  /api/employees/{login_id}
```

### 4. Start the attendance API

Open another terminal:

```bash
cd FaceDetection_Merged
source venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Check:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/hubble/status
```

### 5. Start the React dashboard

Open another terminal:

```bash
cd FaceDetection_Merged/hr-dashboard
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Hubble employee creation

The existing Project_R employee workflow is preserved.

- If **Hubble Login ID is empty**, the employee is created using the normal Project_R manual workflow.
- If **Hubble Login ID is supplied**, the backend calls the Hubble mock API first.
- Hubble supplies employee identity and organizational information.
- The local SQLite attendance record stores the Hubble metadata.
- Face enrollment then continues through the existing Project_R flow.

Example login ID from the supplied mock data:

```text
mbhasyam
```

## Important camera configuration

The current Project_R configuration remains:

```text
/dev/video0 -> ENTRY
/dev/video2 -> EXIT
```

Change these in `api.py` / recognition configuration if the physical camera mapping changes.

## Security

Do not upload real `.env` files or database credentials to GitHub. The repository should contain only `.env.example` / `.env.hubble.example` templates.
