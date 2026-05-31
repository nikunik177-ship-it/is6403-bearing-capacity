"""
Main Application Interface - FastAPI Web Server
IS 6403:1981 Geotechnical Bearing Capacity Estimator
Wraps the analytical engine in a web interface for Render deployment.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import base64
import io
import os

from soil_profile import resolve_shear_failure_mode
from bearing_calculator import AnalyticalBearingEngine

app = FastAPI(title="IS 6403 Bearing Capacity Estimator")

# Inline HTML template (no separate templates folder needed)
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IS 6403:1981 Bearing Capacity Estimator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            color: #e2e8f0;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header .badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.2);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: #a5b4fc;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0.35rem 1rem;
            border-radius: 50px;
            margin-bottom: 1rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #818cf8, #c4b5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
            margin-bottom: 0.75rem;
        }

        .header p {
            color: #94a3b8;
            font-size: 1rem;
            max-width: 500px;
            margin: 0 auto;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1.25rem;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }

        .card h2 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #c4b5fd;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.82rem;
            font-weight: 500;
            color: #94a3b8;
            letter-spacing: 0.02em;
        }

        input[type="number"], select {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 0.6rem;
            padding: 0.65rem 1rem;
            color: #e2e8f0;
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
            width: 100%;
            transition: border-color 0.2s, background 0.2s;
            outline: none;
        }

        input[type="number"]:focus, select:focus {
            border-color: #818cf8;
            background: rgba(129, 140, 248, 0.1);
        }

        select option { background: #1e1b4b; color: #e2e8f0; }

        .hint {
            font-size: 0.75rem;
            color: #64748b;
        }

        .btn {
            width: 100%;
            padding: 0.9rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
            border-radius: 0.75rem;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            margin-top: 0.5rem;
            transition: opacity 0.2s, transform 0.15s;
            letter-spacing: 0.02em;
        }

        .btn:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .result-card {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 0.85rem;
            padding: 1.25rem;
            text-align: center;
        }

        .result-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #a5b4fc;
            line-height: 1;
            margin-bottom: 0.4rem;
        }

        .result-card .label {
            font-size: 0.78rem;
            color: #94a3b8;
            font-weight: 500;
        }

        .result-card .unit {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.2rem;
        }

        .factors-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }

        .factors-table th {
            text-align: left;
            padding: 0.6rem 0.8rem;
            color: #94a3b8;
            font-weight: 500;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }

        .factors-table td {
            padding: 0.6rem 0.8rem;
            color: #c4b5fd;
            font-weight: 600;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }

        .failure-badge {
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .failure-general {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #86efac;
        }

        .failure-local {
            background: rgba(251, 146, 60, 0.15);
            border: 1px solid rgba(251, 146, 60, 0.3);
            color: #fdba74;
        }

        .chart-img {
            width: 100%;
            border-radius: 0.75rem;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            color: #fca5a5;
            font-size: 0.9rem;
        }

        .section-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <div class="badge">🏗️ Geotechnical Engineering</div>
        <h1>IS 6403:1981 Bearing Capacity<br>Estimator</h1>
        <p>Shallow Foundation Analysis · Net &amp; Gross Safe Bearing Capacity · Water Table Correction</p>
    </div>

    {% if error %}
    <div class="error-box" style="margin-bottom:1.5rem;">
        ⚠️ <strong>Error:</strong> {{ error }}
    </div>
    {% endif %}

    <form method="post" action="/calculate">
        <div class="card">
            <h2>⚙️ Soil Parameters</h2>
            <div class="grid">
                <div class="field">
                    <label>Cohesion c (kN/m²)</label>
                    <input type="number" id="c_val" name="c_val" step="0.1" min="0" value="{{ form.c_val if form else '20' }}" required>
                    <span class="hint">Effective cohesion of soil</span>
                </div>
                <div class="field">
                    <label>Friction Angle φ (degrees)</label>
                    <input type="number" id="phi_val" name="phi_val" step="0.1" min="0" max="45" value="{{ form.phi_val if form else '25' }}" required>
                    <span class="hint">Internal friction angle (0–45°)</span>
                </div>
                <div class="field">
                    <label>Unit Weight γ (kN/m³)</label>
                    <input type="number" id="gamma_val" name="gamma_val" step="0.1" min="1" value="{{ form.gamma_val if form else '18' }}" required>
                    <span class="hint">Bulk unit weight of soil</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📐 Foundation Geometry</h2>
            <div class="grid">
                <div class="field">
                    <label>Width B (meters)</label>
                    <input type="number" id="b_val" name="b_val" step="0.1" min="0.1" value="{{ form.b_val if form else '2' }}" required>
                    <span class="hint">Foundation base width</span>
                </div>
                <div class="field">
                    <label>Depth Df (meters)</label>
                    <input type="number" id="df_val" name="df_val" step="0.1" min="0" value="{{ form.df_val if form else '1.5' }}" required>
                    <span class="hint">Founding depth below GL</span>
                </div>
                <div class="field">
                    <label>Shape</label>
                    <select id="shape" name="shape">
                        <option value="square" {% if form and form.shape == 'square' %}selected{% endif %}>Square</option>
                        <option value="strip" {% if form and form.shape == 'strip' %}selected{% endif %}>Strip</option>
                        <option value="circular" {% if form and form.shape == 'circular' %}selected{% endif %}>Circular</option>
                        <option value="rectangular" {% if form and form.shape == 'rectangular' %}selected{% endif %}>Rectangular</option>
                    </select>
                    <span class="hint">Foundation shape type</span>
                </div>
                <div class="field">
                    <label>Water Table Depth (meters)</label>
                    <input type="number" id="dw_val" name="dw_val" step="0.1" min="0" value="{{ form.dw_val if form else '3' }}" required>
                    <span class="hint">Depth of WT below ground level</span>
                </div>
            </div>
        </div>

        <button type="submit" class="btn">⚡ Calculate Bearing Capacity</button>
    </form>

    {% if results %}
    <div class="card" style="margin-top:1.5rem;">
        <h2>📊 Analysis Results</h2>

        <div style="margin-bottom:1.25rem; display:flex; align-items:center; gap:0.75rem;">
            <span class="section-title" style="margin:0;">Failure Mode:</span>
            <span class="failure-badge {{ 'failure-general' if 'General' in results.failure_type else 'failure-local' }}">
                {{ results.failure_type }}
            </span>
        </div>

        <div class="results-grid" style="margin-bottom:1.5rem;">
            <div class="result-card">
                <div class="value">{{ results.q_net_ultimate }}</div>
                <div class="label">Net Ultimate Capacity (q_nu)</div>
                <div class="unit">kN/m²</div>
            </div>
            <div class="result-card">
                <div class="value">{{ results.q_net_safe }}</div>
                <div class="label">Net Safe Capacity (q_ns)</div>
                <div class="unit">kN/m² · FOS = 3.0</div>
            </div>
            <div class="result-card">
                <div class="value">{{ results.q_gross_safe }}</div>
                <div class="label">Gross Safe Capacity</div>
                <div class="unit">kN/m²</div>
            </div>
        </div>

        <div class="section-title">Bearing Capacity Factors (IS 6403 Table 1)</div>
        <table class="factors-table" style="margin-bottom:1.5rem;">
            <tr>
                <th>Factor</th><th>Nc</th><th>Nq</th><th>Nγ</th>
                <th>sc</th><th>sq</th><th>sγ</th>
            </tr>
            <tr>
                <td>Value</td>
                <td>{{ results.nc }}</td>
                <td>{{ results.nq }}</td>
                <td>{{ results.ng }}</td>
                <td>{{ results.sc }}</td>
                <td>{{ results.sq }}</td>
                <td>{{ results.sg }}</td>
            </tr>
        </table>

        {% if results.chart %}
        <div class="section-title">Bearing Capacity vs Foundation Width (NumPy Parametric Sweep)</div>
        <img class="chart-img" src="data:image/png;base64,{{ results.chart }}" alt="Bearing Capacity Chart">
        {% endif %}
    </div>
    {% endif %}

</div>
</body>
</html>
"""

from jinja2 import Environment

jinja_env = Environment()
template = jinja_env.from_string(HTML_PAGE)


@app.get("/", response_class=HTMLResponse)
async def index():
    html = template.render(results=None, error=None, form=None)
    return HTMLResponse(content=html)


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    c_val: float = Form(...),
    phi_val: float = Form(...),
    gamma_val: float = Form(...),
    b_val: float = Form(...),
    df_val: float = Form(...),
    dw_val: float = Form(...),
    shape: str = Form("square"),
):
    form_data = {
        "c_val": c_val,
        "phi_val": phi_val,
        "gamma_val": gamma_val,
        "b_val": b_val,
        "df_val": df_val,
        "dw_val": dw_val,
        "shape": shape,
    }

    try:
        # 1. Resolve failure mode
        soil_profile = resolve_shear_failure_mode(phi_val, c_val, mode="auto")

        # 2. Main capacity calculation
        engine = AnalyticalBearingEngine(depth_df=df_val, width_b=b_val, shape=shape, fos=3.0)
        res = engine.calculate_capacity(gamma_val, soil_profile, dw_val)

        # 3. NumPy parametric sweep: widths 1.0m → 5.0m
        width_matrix = np.linspace(1.0, 5.0, 9)
        safe_caps = []
        for w in width_matrix:
            e = AnalyticalBearingEngine(depth_df=df_val, width_b=w, shape=shape, fos=3.0)
            r = e.calculate_capacity(gamma_val, soil_profile, dw_val)
            safe_caps.append(r["q_net_safe"])

        # 4. Generate matplotlib chart → base64
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#1e1b4b")
        ax.set_facecolor("#16123a")
        ax.plot(width_matrix, safe_caps, marker="s", color="#818cf8",
                linestyle="--", linewidth=2.5, markersize=7, label="q_ns vs B")
        ax.axvline(b_val, color="#c4b5fd", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Current B={b_val}m")
        ax.set_title("Net Safe Bearing Capacity vs Foundation Width", color="#e2e8f0", fontsize=13, pad=12)
        ax.set_xlabel("Foundation Width B (m)", color="#94a3b8")
        ax.set_ylabel("Net Safe Capacity (kN/m²)", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.spines["bottom"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", color="#334155", alpha=0.7)
        ax.legend(facecolor="#1e1b4b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        results = {
            "failure_type": soil_profile["failure_type"],
            "q_net_ultimate": res["q_net_ultimate"],
            "q_net_safe": res["q_net_safe"],
            "q_gross_safe": res["q_gross_safe"],
            "nc": res["factors"][0],
            "nq": res["factors"][1],
            "ng": res["factors"][2],
            "sc": res["corrections"][0],
            "sq": res["corrections"][1],
            "sg": res["corrections"][2],
            "chart": chart_b64,
        }

        html = template.render(results=results, error=None, form=form_data)
        return HTMLResponse(content=html)

    except ValueError as e:
        html = template.render(results=None, error=str(e), form=form_data)
        return HTMLResponse(content=html)
