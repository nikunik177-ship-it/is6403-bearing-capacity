"""
app/main.py
===========
FastAPI application entry point for the IS 6403:1981 Soil Bearing
Capacity Web Estimator.

Routes
------
GET  /                    Serve the frontend HTML dashboard
POST /api/calculate       Run IS 6403 calculation; return JSON
POST /api/report          Run calculation and stream a PDF report
GET  /api/health          Health check endpoint
"""

from __future__ import annotations

import json
from pathlib import Path

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request
# pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates

from app.core.is_6403_engine import calculate_bearing_capacity
from app.schemas.soil_inputs import CalculationResponse, SoilInput
from app.services.pdf_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="IS 6403:1981 Soil Bearing Capacity Estimator",
    description=(
        "A full-stack web application for computing the safe bearing "
        "capacity of shallow foundations per IS 6403:1981 and IS 1904:1986."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Jinja2 template engine
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Static files (CSS, JS, images)
_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request) -> HTMLResponse:
    """
    Serve the main frontend dashboard HTML page.

    Returns
    -------
    HTMLResponse
        Rendered ``index.html`` template.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@app.get("/api/health")
async def health_check() -> dict:
    """
    Simple health check endpoint.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok", "service": "IS 6403:1981 Bearing Capacity API"}


@app.post("/api/calculate", response_model=CalculationResponse)
async def calculate(payload: SoilInput) -> JSONResponse:
    """
    Execute IS 6403:1981 bearing capacity calculation.

    Parameters
    ----------
    payload : SoilInput
        Pydantic-validated site and footing parameters.

    Returns
    -------
    JSONResponse
        Full result dict including all intermediate factors and
        final bearing capacities in kPa.

    Raises
    ------
    HTTPException 422
        Automatic Pydantic validation error response.
    HTTPException 500
        Internal calculation error.
    """
    try:
        results = calculate_bearing_capacity(payload.to_engine_dict())
        return JSONResponse(content=results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal calculation error: {exc}",
        ) from exc


@app.post("/api/report")
async def download_report(payload: SoilInput) -> StreamingResponse:
    """
    Execute IS 6403:1981 calculation and return a downloadable PDF report.

    Parameters
    ----------
    payload : SoilInput
        Pydantic-validated site and footing parameters.

    Returns
    -------
    StreamingResponse
        PDF byte stream with ``Content-Disposition: attachment``.

    Raises
    ------
    HTTPException 500
        PDF generation error.
    """
    try:
        inputs_dict  = payload.to_engine_dict()
        results_dict = calculate_bearing_capacity(inputs_dict)
        pdf_bytes    = generate_pdf_report(inputs_dict, results_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation error: {exc}",
        ) from exc

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                'attachment; filename="IS6403_Bearing_Capacity_Report.pdf"'
            )
        },
    )
