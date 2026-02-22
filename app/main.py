from fastapi import FastAPI
from app.controllers import embedding_controller

app = FastAPI()

app.include_router(embedding_controller.router)

@app.get("/")
def root():
    return {"message": "JobSwipe AI Service running"}