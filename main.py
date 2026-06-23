import os
import uuid
import logging
from logging_config import setup_logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from agent import run_agent
from dotenv import load_dotenv
from datetime import date

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────
ENV = os.getenv("ENV", "development")
VALID_API_KEY = os.getenv("API_KEY", "dev-key-123")

# ── RATE LIMITER ───────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# logger (just define, DO NOT setup yet)
logger = logging.getLogger(__name__)

# ── LIFESPAN ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Server starting...")
    yield

# ── APP ────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Agent API",
    description="Autonomous company research API. Submit a company name and receive a structured research report powered by AI.",
    version="1.0.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── API KEY AUTH ───────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key

# ── REQUEST / RESPONSE MODELS ──────────────────────────
class ResearchRequest(BaseModel):
    company: str

class ResearchResponse(BaseModel):
    company: str
    report: str
    timestamp: str

# ── ROUTES ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "AI Research Agent API is running",
        "usage": "POST /research with {'company': 'company name'}",
        "auth": "Include X-API-Key header with your API key"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/research", response_model=ResearchResponse)
@limiter.limit("10/hour")
async def research(
    request: Request,
    body: ResearchRequest,
    api_key: str = Security(verify_api_key)
):
    if not body.company.strip():
        raise HTTPException(
            status_code=400,
            detail="Company name cannot be empty"
        )

    logger.info("Research request received: %s", body.company)

    report = run_agent(body.company)

    return ResearchResponse(
        company=body.company,
        report=report,
        timestamp=str(date.today())
    )