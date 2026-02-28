# ==============================================================
# FILE: titanic_agent/backend/main.py
# PURPOSE: FastAPI application — POST /chat endpoint and lifecycle.
# ==============================================================

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agent import TitanicAgent
from config import settings
from data_loader import DatasetValidationError, get_dataframe

# --------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Application state
# --------------------------------------------------------------

_agent: Optional[TitanicAgent] = None


# --------------------------------------------------------------
# Lifespan: warm-up dataset and agent before accepting requests
# --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent

    logger.info("Server starting — loading Titanic dataset...")
    try:
        get_dataframe()
        logger.info("Dataset loaded and cached successfully.")
    except DatasetValidationError as exc:
        logger.critical("Dataset validation failed on startup: %s", exc)
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        logger.critical("Unexpected error loading dataset: %s", exc)
        raise

    logger.info("Initialising LangChain agent...")
    _agent = TitanicAgent()
    logger.info("Agent ready. Server is accepting requests.")

    yield

    logger.info("Server shutting down.")


# --------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------

app = FastAPI(
    title="Titanic Chat Agent API",
    description=(
        "A LangChain-powered agent that answers natural language questions "
        "about the Titanic passenger dataset and returns charts on demand."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --------------------------------------------------------------
# Request / Response schemas
# --------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="A natural language question about the Titanic dataset.",
        examples=["What percentage of passengers were male?"],
    )

    @field_validator("question")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank.")
        return value


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Text answer produced by the agent.")
    has_chart: bool = Field(..., description="True if a chart was generated for this response.")
    chart_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG image string, present only when has_chart is true.",
    )
    processing_time_ms: float = Field(
        ..., description="Total server-side processing time in milliseconds."
    )


class ErrorResponse(BaseModel):
    detail: str


# --------------------------------------------------------------
# Middleware: per-request access logging
# --------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------

@app.get(
    "/health",
    tags=["Utility"],
    summary="Health check and dataset status.",
)
async def health_check():
    """
    Returns server health, loaded dataset row count, and configured LLM model.
    Use this to verify the server is ready before sending chat requests.
    """
    df = get_dataframe()
    return {
        "status": "ok",
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "model": settings.GROQ_MODEL,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Agent not yet initialised."},
        422: {"model": ErrorResponse, "description": "Invalid request payload."},
        500: {"model": ErrorResponse, "description": "Internal agent error."},
    },
    tags=["Chat"],
    summary="Submit a natural language question about the Titanic dataset.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accepts a natural language question, runs it through the LangChain agent,
    and returns a text answer plus an optional base64-encoded chart image.
    """
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not yet initialised. Please retry in a few seconds.",
        )

    logger.info("Incoming question: %r", request.question)
    t_start = time.perf_counter()

    try:
        answer, chart_b64 = _agent.run(request.question)
    except Exception as exc:
        logger.exception("Unhandled error during agent.run(): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again.",
        ) from exc

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    logger.info(
        "Response ready. has_chart=%s, processing_time=%.1f ms",
        chart_b64 is not None,
        elapsed_ms,
    )

    return ChatResponse(
        answer=answer,
        has_chart=chart_b64 is not None,
        chart_base64=chart_b64,
        processing_time_ms=elapsed_ms,
    )


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=False,
        log_level="info",
        access_log=False,  # handled by middleware above
    )
