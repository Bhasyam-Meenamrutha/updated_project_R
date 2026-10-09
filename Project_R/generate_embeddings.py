import os
import cv2
import pickle
import numpy as np

from insightface.app import FaceAnalysis


# ==========================================
# CONFIGURATION
# ==========================================

FACES_DIRECTORY = "data/faces_v2"
EMBEDDINGS_DIRECTORY = "data/embeddings"

EMBEDDINGS_FILE = os.path.join(
    EMBEDDINGS_DIRECTORY,
    "face_embeddings.pkl"
)


# ==========================================
# CREATE EMBEDDINGS DIRECTORY
# ==========================================

os.makedirs(
    EMBEDDINGS_DIRECTORY,
    exist_ok=True
)


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print()
print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l"
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")
print()


# ==========================================
# CHECK FACE DATA
# ==========================================

if not os.path.exists(FACES_DIRECTORY):

    print(
        "ERROR: Face data directory does not exist."
    )

    exit()


# ==========================================
# STORAGE
# ==========================================

employee_embeddings = {}


# ==========================================
# PROCESS EMPLOYEE FOLDERS
# ==========================================

employee_folders = sorted(
    os.listdir(FACES_DIRECTORY)
)


for employee_id in employee_folders:

    employee_path = os.path.join(
        FACES_DIRECTORY,
        employee_id
    )


    if not os.path.isdir(employee_path):
        continue


    print(
        f"Processing employee: {employee_id}"
    )


    embeddings = []


    # --------------------------------------
    # PROCESS EMPLOYEE IMAGES
    # --------------------------------------

    image_files = sorted(
        os.listdir(employee_path)
    )


    for image_name in image_files:

        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue


        image_path = os.path.join(
            employee_path,
            image_name
        )


        image = cv2.imread(
            image_path
        )


        if image is None:

            print(
                f"Could not read: {image_path}"
            )

            continue


        # ----------------------------------
        # DETECT FACE
        # ----------------------------------

        faces = app.get(
            image
        )


        if len(faces) == 0:

            print(
                f"No face detected: "
                f"{image_name}"
            )

            continue


        # ----------------------------------
        # SELECT LARGEST FACE
        # ----------------------------------

        face = max(
            faces,
            key=lambda detected_face:
            (
                detected_face.bbox[2]
                - detected_face.bbox[0]
            )
            *
            (
                detected_face.bbox[3]
                - detected_face.bbox[1]
            )
        )


        # ----------------------------------
        # GET EMBEDDING
        # ----------------------------------

        embedding = face.normed_embedding


        if embedding is None:

            print(
                f"Could not generate embedding: "
                f"{image_name}"
            )

            continue


        # ----------------------------------
        # STORE EMBEDDING
        # ----------------------------------

        embeddings.append(
            embedding
        )


        print(
            f"  ✓ {image_name}"
        )


    # ======================================
    # CHECK EMPLOYEE EMBEDDINGS
    # ======================================

    if len(embeddings) == 0:

        print(
            f"No valid embeddings for "
            f"employee {employee_id}"
        )

        continue


    # ======================================
    # CALCULATE AVERAGE EMBEDDING
    # ======================================

    embeddings_array = np.array(
        embeddings
    )


    average_embedding = np.mean(
        embeddings_array,
        axis=0
    )


    # ======================================
    # NORMALIZE AVERAGE EMBEDDING
    # ======================================

    norm = np.linalg.norm(
        average_embedding
    )


    if norm == 0:

        print(
            f"Invalid embedding for "
            f"{employee_id}"
        )

        continue


    average_embedding = (
        average_embedding / norm
    )


    # ======================================
    # SAVE EMPLOYEE EMBEDDING
    # ======================================

    employee_embeddings[
        employee_id
    ] = average_embedding


    print(
        f"Employee {employee_id}: "
        f"{len(embeddings)} embeddings created."
    )


    print()


# ==========================================
# CHECK FINAL DATA
# ==========================================

if len(employee_embeddings) == 0:

    print(
        "ERROR: No employee embeddings "
        "were generated."
    )

    exit()


# ==========================================
# SAVE TO FILE
# ==========================================

with open(
    EMBEDDINGS_FILE,
    "wb"
) as file:

    pickle.dump(
        employee_embeddings,
        file
    )


# ==========================================
# FINAL RESULT
# ==========================================

print()
print("========================================")
print("EMBEDDING GENERATION COMPLETED")
print("========================================")

print()

print(
    f"Employees processed: "
    f"{len(employee_embeddings)}"
)

print()

for employee_id, embedding in (
    employee_embeddings.items()
):

    print(
        f"Employee {employee_id}: "
        f"Embedding size = {len(embedding)}"
    )


print()

print(
    f"Saved embeddings to:"
)

print(
    EMBEDDINGS_FILE
)

print()