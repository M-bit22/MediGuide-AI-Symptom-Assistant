import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]

GENDERS = ["Female", "Male", "Other", "Prefer not to say"]

COMMON_SYMPTOMS = [
    "Fever", "Cough", "Headache", "Fatigue", "Shortness of Breath",
    "Chest Pain", "Sore Throat", "Runny Nose", "Nausea", "Dizziness",
    "Joint Pain", "Stomach Ache", "Rash"
]

DURATIONS = [
    "Less than 24 hours",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "More than 2 weeks"
]

LANGUAGES = ["English", "Urdu", "Spanish", "French", "Arabic"]