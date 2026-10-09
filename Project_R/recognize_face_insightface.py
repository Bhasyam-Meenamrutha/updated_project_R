import cv2
import os
import time
import pickle
import argparse
import numpy as np

from collections import defaultdict, deque
from insightface.app import FaceAnalysis

from attendance import record_entry, record_exit


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="InsightFace Employee Attendance Recognition"
)

parser.add_argument(
    "--camera",
    required=True,
    help="Camera device path, e.g. /dev/video0 or /dev/video2"
)

parser.add_argument(
    "--mode",
    required=True,
    choices=["ENTRY", "EXIT"],
    help="Fixed camera responsibility"
)

args = parser.parse_args()

CAMERA_DEVICE = args.camera
CAMERA_MODE = args.mode


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "embeddings",
    "face_embeddings.pkl"
)

# Recognition threshold
SIMILARITY_THRESHOLD = 0.45

# Minimum difference between best and second-best employee
CONFIDENCE_MARGIN = 0.05

# InsightFace detector
DETECTION_SIZE = (320, 320)

# Process every Nth frame
PROCESS_EVERY_N_FRAMES = 5

# Stable recognition
HISTORY_SIZE = 5
REQUIRED_MATCHES = 3

# Attendance protection
ATTENDANCE_COOLDOWN = 5

# Tracking
TRACK_DISTANCE_THRESHOLD = 120
TRACK_TIMEOUT = 2.0


# ============================================================
# LIVENESS / ANTI-SPOOFING
# ============================================================

LIVENESS_THRESHOLD = 0.80

# Number of live results required before allowing recognition
LIVENESS_HISTORY_SIZE = 3
LIVENESS_REQUIRED = 2


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

RECOGNITION_WIDTH = 320
RECOGNITION_HEIGHT = 240


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = f"Face Recognition - {CAMERA_MODE}"


# ============================================================
# TRACKING
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)

track_last_seen = {}

track_last_position = {}

attendance_cooldown = {}

# Liveness history for each tracked face
liveness_history = defaultdict(
    lambda: deque(maxlen=LIVENESS_HISTORY_SIZE)
)


# ============================================================
# EMBEDDINGS
# ============================================================

employee_embeddings = {}

embeddings_modified_time = 0


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(force=False):

    global employee_embeddings
    global embeddings_modified_time

    if not os.path.exists(EMBEDDINGS_FILE):

        print()
        print("ERROR: Embeddings file not found:")
        print(EMBEDDINGS_FILE)
        print()

        return False

    try:

        modified_time = os.path.getmtime(
            EMBEDDINGS_FILE
        )

        if (
            not force
            and
            modified_time == embeddings_modified_time
        ):
            return True

        with open(
            EMBEDDINGS_FILE,
            "rb"
        ) as file:

            loaded_embeddings = pickle.load(
                file
            )

        if not isinstance(
            loaded_embeddings,
            dict
        ):

            print(
                "ERROR: Invalid embeddings format."
            )

            return False

        new_embeddings = {}

        for employee_id, embeddings in loaded_embeddings.items():

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            if embeddings.ndim == 1:

                embeddings = embeddings.reshape(
                    1,
                    -1
                )

            if embeddings.size == 0:
                continue

            new_embeddings[
                str(employee_id)
            ] = embeddings

        employee_embeddings = new_embeddings

        embeddings_modified_time = modified_time

        print()
        print(
            "Employee embeddings loaded:"
        )
        print(
            "---------------------------"
        )

        for employee_id, embeddings in employee_embeddings.items():

            print(
                f"Employee {employee_id}: "
                f"{embeddings.shape}"
            )

        print()

        return True

    except Exception as e:

        print(
            "ERROR loading embeddings:",
            e
        )

        return False


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):

    denominator = (
        np.linalg.norm(embedding_a)
        *
        np.linalg.norm(embedding_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            embedding_a,
            embedding_b
        )
        /
        denominator
    )


# ============================================================
# FIND EMPLOYEE
# ============================================================

def find_employee(face_embedding):

    employee_scores = {}

    for employee_id, embeddings in employee_embeddings.items():

        best_employee_score = -1.0

        for stored_embedding in embeddings:

            score = cosine_similarity(
                face_embedding,
                stored_embedding
            )

            if score > best_employee_score:

                best_employee_score = score

        employee_scores[
            employee_id
        ] = best_employee_score

    if not employee_scores:

        return None, 0.0, 0.0

    sorted_scores = sorted(
        employee_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_employee = sorted_scores[0][0]

    best_score = sorted_scores[0][1]

    if len(sorted_scores) > 1:

        second_best_score = sorted_scores[1][1]

    else:

        second_best_score = 0.0

    confidence_margin = (
        best_score
        -
        second_best_score
    )

    if best_score < SIMILARITY_THRESHOLD:

        return (
            None,
            best_score,
            confidence_margin
        )

    if (
        len(sorted_scores) > 1
        and
        confidence_margin < CONFIDENCE_MARGIN
    ):

        print(
            f"AMBIGUOUS FACE | "
            f"Best={best_employee} "
            f"{best_score:.3f} | "
            f"Second={sorted_scores[1][0]} "
            f"{second_best_score:.3f} | "
            f"Margin={confidence_margin:.3f}"
        )

        return (
            None,
            best_score,
            confidence_margin
        )

    return (
        best_employee,
        best_score,
        confidence_margin
    )


# ============================================================
# RECORD ATTENDANCE
# ============================================================

def record_attendance_event(employee_id):

    current_time = time.time()

    cooldown_key = (
        str(employee_id),
        CAMERA_MODE
    )

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if cooldown_key in attendance_cooldown:

        elapsed = (
            current_time
            -
            attendance_cooldown[
                cooldown_key
            ]
        )

        if elapsed < ATTENDANCE_COOLDOWN:

            return

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if CAMERA_MODE == "ENTRY":

        result = record_entry(
            str(employee_id)
        )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    else:

        result = record_exit(
            str(employee_id)
        )

    attendance_cooldown[
        cooldown_key
    ] = current_time

    print()
    print(
        "========================================"
    )

    print(
        f"[{CAMERA_MODE}] "
        f"Employee {employee_id}"
    )

    print(result)

    print(
        "========================================"
    )


# ============================================================
# TRACK ID
# ============================================================

def get_track_id(
    center_x,
    center_y
):

    current_time = time.time()

    current_position = (
        center_x,
        center_y
    )

    # --------------------------------------------------------
    # Remove expired tracks
    # --------------------------------------------------------

    expired_tracks = []

    for track_id, last_seen in track_last_seen.items():

        if (
            current_time
            -
            last_seen
            >
            TRACK_TIMEOUT
        ):

            expired_tracks.append(
                track_id
            )

    for track_id in expired_tracks:

        track_last_seen.pop(
            track_id,
            None
        )

        track_last_position.pop(
            track_id,
            None
        )

        track_history.pop(
            track_id,
            None
        )

        liveness_history.pop(
            track_id,
            None
        )

    # --------------------------------------------------------
    # Find nearest existing track
    # --------------------------------------------------------

    best_track = None

    best_distance = float("inf")

    for track_id, position in track_last_position.items():

        distance = np.linalg.norm(
            np.array(current_position)
            -
            np.array(position)
        )

        if (
            distance < best_distance
            and
            distance < TRACK_DISTANCE_THRESHOLD
        ):

            best_distance = distance
            best_track = track_id

    # --------------------------------------------------------
    # Existing track
    # --------------------------------------------------------

    if best_track is not None:

        track_id = best_track

    # --------------------------------------------------------
    # New track
    # --------------------------------------------------------

    else:

        track_id = (
            f"{current_time:.6f}_"
            f"{center_x}_"
            f"{center_y}"
        )

    track_last_seen[
        track_id
    ] = current_time

    track_last_position[
        track_id
    ] = current_position

    return track_id


# ============================================================
# LIVENESS CHECK
# ============================================================

def check_liveness(
    face,
    track_id
):

    liveness = getattr(
        face,
        "liveness",
        None
    )

    # --------------------------------------------------------
    # No liveness information
    # --------------------------------------------------------

    if liveness is None:

        print(
            "LIVENESS ERROR | "
            "No liveness information"
        )

        liveness_history[
            track_id
        ].clear()

        return (
            False,
            0.0,
            "LIVENESS ERROR"
        )

    # --------------------------------------------------------
    # InsightFace liveness result
    # --------------------------------------------------------

    if isinstance(
        liveness,
        dict
    ):

        status = liveness.get(
            "status"
        )

        is_live = bool(
            liveness.get(
                "is_live",
                False
            )
        )

        live_score = float(
            liveness.get(
                "live_score",
                0.0
            )
        )

    else:

        status = getattr(
            liveness,
            "status",
            None
        )

        is_live = bool(
            getattr(
                liveness,
                "is_live",
                False
            )
        )

        live_score = float(
            getattr(
                liveness,
                "live_score",
                0.0
            )
        )

    # --------------------------------------------------------
    # Invalid/rejected input
    # --------------------------------------------------------

    if status != "ok":

        liveness_history[
            track_id
        ].clear()

        return (
            False,
            live_score,
            "SPOOF / REJECTED"
        )

    # --------------------------------------------------------
    # Score-based live decision
    # --------------------------------------------------------

    live_result = (
        is_live
        and
        live_score >= LIVENESS_THRESHOLD
    )

    liveness_history[
        track_id
    ].append(
        live_result
    )

    live_count = sum(
        liveness_history[
            track_id
        ]
    )

    # --------------------------------------------------------
    # Require stable liveness
    # --------------------------------------------------------

    if live_count >= LIVENESS_REQUIRED:

        return (
            True,
            live_score,
            "LIVE"
        )

    # --------------------------------------------------------
    # Not yet stable
    # --------------------------------------------------------

    if not live_result:

        return (
            False,
            live_score,
            "SPOOF"
        )

    return (
        False,
        live_score,
        "VERIFYING LIVE"
    )


# ============================================================
# DRAW BOX AND LABEL
# ============================================================

def draw_face_box(
    display_frame,
    x1,
    y1,
    x2,
    y2,
    color,
    label
):

    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label_height = 30

    label_y1 = max(
        0,
        y1 - label_height
    )

    cv2.rectangle(
        display_frame,
        (x1, label_y1),
        (x2, y1),
        color,
        -1
    )

    cv2.putText(
        display_frame,
        label,
        (
            x1 + 5,
            max(
                20,
                y1 - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
        2
    )


# ============================================================
# PROCESS FACES
# ============================================================

def process_faces(
    display_frame,
    faces,
    scale_x,
    scale_y
):

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # ----------------------------------------------------
        # Convert recognition coordinates
        # ----------------------------------------------------

        x1 = int(
            x1 * scale_x
        )

        y1 = int(
            y1 * scale_y
        )

        x2 = int(
            x2 * scale_x
        )

        y2 = int(
            y2 * scale_y
        )

        # ----------------------------------------------------
        # Keep inside frame
        # ----------------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            display_frame.shape[1] - 1,
            x2
        )

        y2 = min(
            display_frame.shape[0] - 1,
            y2
        )

        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        track_id = get_track_id(
            center_x,
            center_y
        )

        # ====================================================
        # LIVENESS CHECK
        # ====================================================

        is_live, live_score, live_status = check_liveness(
            face,
            track_id
        )

        # ====================================================
        # SPOOF / NOT VERIFIED
        # ====================================================

        if not is_live:

            # Absolutely clear recognition history.
            # A spoof must never inherit previous recognition.
            track_history[
                track_id
            ].clear()

            # ------------------------------------------------
            # RED BOX
            # ------------------------------------------------

            color = (
                0,
                0,
                255
            )

            if live_status == "VERIFYING LIVE":

                label = (
                    f"VERIFYING "
                    f"{live_score:.2f}"
                )

            elif live_status == "LIVENESS ERROR":

                label = "LIVENESS ERROR"

            else:

                label = (
                    f"SPOOF "
                    f"{live_score:.2f}"
                )

            draw_face_box(
                display_frame,
                x1,
                y1,
                x2,
                y2,
                color,
                label
            )

            # Do NOT access face.embedding.
            # Do NOT recognize.
            # Do NOT record attendance.

            continue

        # ====================================================
        # REAL PERSON
        # ====================================================

        # At this point liveness has passed.
        # Recognition can safely continue.

        if not hasattr(
            face,
            "embedding"
        ):

            draw_face_box(
                display_frame,
                x1,
                y1,
                x2,
                y2,
                (
                    0,
                    0,
                    255
                ),
                "NO EMBEDDING"
            )

            track_history[
                track_id
            ].clear()

            continue

        # ----------------------------------------------------
        # Recognition
        # ----------------------------------------------------

        employee_id, score, margin = find_employee(
            face.embedding
        )

        # ====================================================
        # UNKNOWN / AMBIGUOUS
        # ====================================================

        if employee_id is None:

            color = (
                0,
                0,
                255
            )

            label = (
                f"UNKNOWN "
                f"{score:.2f}"
            )

            # Unknown face must not build history.
            track_history[
                track_id
            ].clear()

        # ====================================================
        # KNOWN REAL EMPLOYEE
        # ====================================================

        else:

            color = (
                0,
                255,
                0
            )

            label = (
                f"{employee_id} "
                f"{score:.2f} "
                f"LIVE"
            )

            # ------------------------------------------------
            # Add recognition result
            # ------------------------------------------------

            track_history[
                track_id
            ].append(
                str(employee_id)
            )

            history = track_history[
                track_id
            ]

            # ------------------------------------------------
            # Count matches
            # ------------------------------------------------

            counts = {}

            for recognized_id in history:

                counts[
                    recognized_id
                ] = (
                    counts.get(
                        recognized_id,
                        0
                    )
                    +
                    1
                )

            best_id = max(
                counts,
                key=counts.get
            )

            best_count = counts[
                best_id
            ]

            # ------------------------------------------------
            # Stable recognition
            # ------------------------------------------------

            if best_count >= REQUIRED_MATCHES:

                record_attendance_event(
                    best_id
                )

                track_history[
                    track_id
                ].clear()

        # ====================================================
        # DRAW BOX
        # ====================================================

        draw_face_box(
            display_frame,
            x1,
            y1,
            x2,
            y2,
            color,
            label
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )

    print(
        "   FACE ATTENDANCE RECOGNITION"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Camera : {CAMERA_DEVICE}"
    )

    print(
        f"Mode   : {CAMERA_MODE}"
    )

    print(
        f"Embeddings : {EMBEDDINGS_FILE}"
    )

    print(
        f"Similarity threshold : "
        f"{SIMILARITY_THRESHOLD}"
    )

    print(
        f"Confidence margin : "
        f"{CONFIDENCE_MARGIN}"
    )

    print(
        f"Liveness threshold : "
        f"{LIVENESS_THRESHOLD}"
    )

    print()

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    if not load_embeddings(
        force=True
    ):

        return

    # ========================================================
    # LOAD INSIGHTFACE WITH LIVENESS
    # ========================================================

    print(
        "Loading InsightFace model "
        "with liveness..."
    )

    app = FaceAnalysis(
        name="buffalo_l",
        addons=[
            "liveness"
        ],
        liveness_mode="normal",
        liveness_threshold=LIVENESS_THRESHOLD
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    print(
        "InsightFace model with "
        "liveness loaded."
    )

    print()

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open camera:"
        )

        print(
            CAMERA_DEVICE
        )

        return

    # ========================================================
    # CAMERA SETTINGS
    # ========================================================

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        DISPLAY_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        DISPLAY_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print(
        f"{CAMERA_MODE} camera started."
    )

    print(
        f"Device: {CAMERA_DEVICE}"
    )

    print(
        "Liveness protection: ENABLED"
    )

    print(
        "Press Q to quit."
    )

    print()

    # ========================================================
    # VARIABLES
    # ========================================================

    frame_count = 0

    # ========================================================
    # SCALE FACTORS
    # ========================================================

    scale_x = (
        DISPLAY_WIDTH
        /
        RECOGNITION_WIDTH
    )

    scale_y = (
        DISPLAY_HEIGHT
        /
        RECOGNITION_HEIGHT
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Could not read frame."
            )

            break

        frame_count += 1

        # ====================================================
        # RESIZE FOR RECOGNITION
        # ====================================================

        recognition_frame = cv2.resize(
            frame,
            (
                RECOGNITION_WIDTH,
                RECOGNITION_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        # ====================================================
        # PROCESS EVERY NTH FRAME
        # ====================================================

        if (
            frame_count
            %
            PROCESS_EVERY_N_FRAMES
            ==
            0
        ):

            try:

                faces = app.get(
                    recognition_frame
                )

                process_faces(
                    frame,
                    faces,
                    scale_x,
                    scale_y
                )

            except Exception as e:

                print(
                    "Face processing error:",
                    e
                )

        # ====================================================
        # CAMERA LABEL
        # ====================================================

        cv2.putText(
            frame,
            f"{CAMERA_MODE} CAMERA",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            "LIVENESS: ON",
            (
                20,
                65
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            frame,
            CAMERA_DEVICE,
            (
                20,
                95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    print()

    print(
        f"{CAMERA_MODE} camera stopped."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()