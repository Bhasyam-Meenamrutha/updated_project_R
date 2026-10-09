import os

from pymongo import MongoClient
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

load_dotenv(
    os.path.join(BASE_DIR, ".env.hubble")
)


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017"
)

DATABASE_NAME = os.getenv(
    "MONGO_DATABASE",
    "hubble_employee_db"
)


client = MongoClient(MONGO_URI)

db = client[DATABASE_NAME]

employees_collection = db["employees"]

employee_directory_collection = db[
    "employee_directory"
]