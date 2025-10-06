from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from validate_license import router as license_router
from validate_selfie import router as face_router
import os
import time
import asyncio
from functools import lru_cache
import json
import logging
import json
from datetime import datetime

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kyc_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Audit logging for compliance
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)
audit_handler = logging.FileHandler('kyc_audit.log')
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - AUDIT - %(message)s'
))
audit_logger.addHandler(audit_handler)
audit_logger.propagate = False

app = FastAPI(
    title="KYC License Validation API",
    description="Secure API for US license validation and face matching",
    version="1.0.0"
)

# Performance monitoring middleware
@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    """Add performance monitoring headers"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    logger.info(f"{request.method} {request.url.path} - Processing time: {process_time:.4f}s")

    return response

# Health check endpoint

# -----------------------------
# CORS setup - Production ready
# -----------------------------
# In production, replace with actual frontend domains
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
