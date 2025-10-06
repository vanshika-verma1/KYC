from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from validate_license import router as license_router
from validate_selfie import router as face_router 

app = FastAPI()

# -----------------------------
# CORS setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Include routers
# -----------------------------
app.include_router(license_router, prefix="/license", tags=["Validate License"])
app.include_router(face_router, prefix="/selfie", tags=["Validate Selfie"])

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
