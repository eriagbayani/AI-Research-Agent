import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
from agent import run_agent
from dotenv import load_dotenv

load_dotenv()

# ── APP ────────────────────────────────────────────────
ENV = os.getenv("ENV", "development")

app = FastAPI(
    title="AI Research Agent API",
    description="Give it a company name — get a structured research report back.",
    version="1.0.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
)

# ── REQUEST / RESPONSE MODELS ──────────────────────────
class ResearchRequest(BaseModel):
    company: str

class ResearchResponse(BaseModel):
    company: str
    report: str
    timestamp: str

# ── ROUTES ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Research Agent API is running",
        "usage": "POST /research with {'company': 'company name'}"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    if not request.company.strip():
        raise HTTPException(
            status_code=400,
            detail="Company name cannot be empty"
        )

    print(f"🔍 Research request received: {request.company}")

    report = run_agent(request.company)

    return ResearchResponse(
        company=request.company,
        report=report,
        timestamp=str(date.today())
    )