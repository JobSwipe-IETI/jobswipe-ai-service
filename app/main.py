from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv
from app.controllers import embedding_controller
from app.controllers import profile_controller

# Try to load environment variables from .env file (optional).
# If running via start.ps1, variables are already set.
try:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except Exception:
    pass

app = FastAPI()

app.include_router(embedding_controller.router)
app.include_router(profile_controller.router)

@app.get("/")
def root():
    return {"message": "JobSwipe AI Service running"}