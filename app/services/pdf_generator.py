"""
app/services/pdf_generator.py
=============================
ReportLab-based generator for a professionally formatted
"Geotechnical Design Calculation Report" in PDF format.

The report includes:
    - Project header with IS code reference
    - Design inputs table
    - Failure mode and effective parameters
    - Bearing Capacity Factors table
    - Correction Factors table (shape / depth / inclination)
    - Water table correction table
    - Step-by-step capacity equation terms
    - Final capacity summary box
    - Footer with disclaimer

Engineering Units: kPa · kN/m³ · m · degrees (SI)
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK_BLUE  = colors.HexColor("#1a237e")
_MID_BLUE   = colors.HexColor("#1565c0")
_LIGHT_BLUE = colors.HexColor("#e3f2fd")
_ACCENT     = colors.HexColor("#0d47a1")
_SUCCESS    = colors.HexColor("#1b5e20")
_SUCCESS_BG = colors.HexColor("#e8f5e9")
_GREY_BG    = colors.HexColor("#f5f5f5")
_GREY_LINE  = colors.HexColor("#bdbdbd")
_WHITE      = colors.white
_BLACK      = colors.black

PAGE_W, PAGE_H = A4


def _styles() -> dict[str, ParagraphStyle]:
    """Build and return the report paragraph style library."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=16,
            textColor=_DARK_BLUE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=_MID_BLUE,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=_WHITE,
            backColor=_MID_BLUE,
            leftIndent=6,
            spaceBefore=12,
            spaceAfter=4,
            leading=16,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=_BLACK,
            spaceAfter=2,
            leading=14,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "result_label": ParagraphStyle(
            "result_label",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=_SUCCESS,
            alignment=TA_LEFT,
        ),
        "result_value": ParagraphStyle(
            "result_value",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=_SUCCESS,
            alignment=TA_RIGHT,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica-Oblique",
            fontSize=7,
            textColor=colors.grey,
            alignment=TA_CENTER,
        ),
        "eq": ParagraphStyle(
            "eq",
            fontName="Courier",
            fontSize=8,
            textColor=_ACCENT,
            backColor=_GREY_BG,
            leftIndent=8,
            spaceAfter=4,
            leading=12,
        ),
    }


# ---------------------------------------------------------------------------
# Table style helpers
# ---------------------------------------------------------------------------

def _header_table_style(num_cols: int) -> TableStyle:
    """Standard style for data tables with a blue header row."""
    cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  _MID_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  _WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, 0),  "CENTER"),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _GREY_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.5, _GREY_LINE),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(cmds)


def _result_box_style() -> TableStyle:
    """Style for the final capacity summary box."""
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _SUCCESS_BG),
        ("TEXTCOLOR",     (0, 0), (-1, -1), _SUCCESS),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("ALIGN",         (1, 0), (1, -1),  "RIGHT"),
        ("BOX",           (0, 0), (-1, -1), 1.5, _SUCCESS),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, colors.HexColor("#a5d6a7")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ])


# ---------------------------------------------------------------------------
# Public generator function
# ---------------------------------------------------------------------------

def generate_pdf_report(
    inputs: dict[str, Any],
    results: dict[str, Any],
) -> bytes:
    """
    Generate a complete Geotechnical Design Calculation Report as PDF bytes.

    Parameters
    ----------
    inputs : dict
        Raw site and footing parameters supplied by the user.
    results : dict
        Output dict returned by ``is_6403_engine.calculate_bearing_capacity()``.

    Returns
    -------
    bytes
        Complete PDF file content, ready for streaming to the browser.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="IS 6403:1981 Geotechnical Design Report",
    )

    s = _styles()
    story: list = []
    col_w = PAGE_W - 3.6 * cm          # usable width

    # ==================================================================
    # 1. Header
    # ==================================================================
    story.append(Paragraph("GEOTECHNICAL DESIGN CALCULATION REPORT", s["title"]))
    story.append(Paragraph(
        "Bearing Capacity of Shallow Foundation  ·  IS 6403:1981 / IS 1904:1986",
        s["subtitle"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=_MID_BLUE))
    story.append(Spacer(1, 0.2 * cm))

    now_str = datetime.now().strftime("%d %B %Y  %H:%M")
    meta = [
        ["Report Generated:", now_str, "Code Reference:", "IS 6403:1981"],
        ["Factor of Safety:", f"{results['fos']}", "Software:", "IS6403 Web Estimator v1.0"],
    ]
    meta_tbl = Table(meta, colWidths=[3.5 * cm, 5.5 * cm, 3.5 * cm, 5 * cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME",  (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), _DARK_BLUE),
        ("TEXTCOLOR", (2, 0), (2, -1), _DARK_BLUE),
        ("GRID",      (0, 0), (-1, -1), 0.3, _GREY_LINE),
        ("BACKGROUND",(0, 0), (-1, -1), _GREY_BG),
        ("TOPPADDING",(0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ==================================================================
    # 2. Design Inputs
    # ==================================================================
    story.append(Paragraph("  1. Design Inputs", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    def _v(key: str, unit: str = "", decimals: int = 2) -> str:
        val = inputs.get(key, results.get(key, "—"))
        try:
            val = f"{float(val):.{decimals}f}"
        except (TypeError, ValueError):
            val = str(val)
        return f"{val} {unit}".strip()

    inp_data = [
        ["Parameter", "Symbol", "Value", "Unit"],
        ["Cohesion",                  "c",       _v("cohesion"),            "kPa"],
        ["Friction Angle",            "φ",       _v("friction_angle"),      "°"],
        ["Bulk Unit Weight",          "γ",       _v("unit_weight"),         "kN/m³"],
        ["Saturated Unit Weight",     "γ_sat",   _v("sat_unit_weight"),     "kN/m³"],
        ["Water Table Depth (GL)",    "dw",
         "N/A" if inputs.get("water_table_depth", 999) >= 999
         else f"{inputs['water_table_depth']:.2f}", "m"],
        ["Footing Shape",             "—",       str(inputs.get("footing_shape","—")).capitalize(), "—"],
        ["Footing Width",             "B",       _v("width"),               "m"],
        ["Footing Length",            "L",
         "N/A" if str(inputs.get("footing_shape","")) != "rectangular"
         else _v("length"), "m"],
        ["Founding Depth",            "Df",      _v("depth"),               "m"],
        ["Load Inclination Angle",    "α",       _v("load_inclination"),    "°"],
        ["Failure Mode (selected)",   "—",       results.get("failure_mode_used","—").upper(), "—"],
    ]
    inp_tbl = Table(inp_data, colWidths=[6.5 * cm, 2 * cm, 4 * cm, 3 * cm])
    inp_tbl.setStyle(_header_table_style(4))
    story.append(inp_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ==================================================================
    # 3. Effective Parameters
    # ==================================================================
    story.append(Paragraph("  2. Effective Shear Parameters (IS 6403 Cl. 5.1.1)", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    if results["failure_mode_used"] == "local":
        eq_text = (
            "Local Shear Failure selected:\n"
            f"  c' = (2/3) × c = (2/3) × {inputs.get('cohesion',0):.2f} = "
            f"{results['c_eff']:.4f} kPa\n"
            f"  φ' = arctan(0.67 × tan φ) = arctan(0.67 × tan "
            f"{inputs.get('friction_angle',0):.1f}°) = {results['phi_eff']:.4f}°"
        )
    else:
        eq_text = (
            "General Shear Failure selected:\n"
            f"  c_eff = c = {results['c_eff']:.4f} kPa\n"
            f"  φ_eff = φ = {results['phi_eff']:.4f}°"
        )
    story.append(Paragraph(eq_text, s["eq"]))
    story.append(Spacer(1, 0.3 * cm))

    # ==================================================================
    # 4. Bearing Capacity Factors
    # ==================================================================
    story.append(Paragraph("  3. Bearing Capacity Factors (IS 6403 Table 1)", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    bc_data = [
        ["Factor", "Symbol", "Value", "Reference"],
        ["Cohesion factor",      "Nc", f"{results['N_c']:.4f}",     "IS 6403 Table 1"],
        ["Surcharge factor",     "Nq", f"{results['N_q']:.4f}",     "IS 6403 Table 1"],
        ["Unit weight factor",   "Nγ", f"{results['N_gamma']:.4f}", "IS 6403 Table 1"],
    ]
    bc_tbl = Table(bc_data, colWidths=[6 * cm, 2.5 * cm, 3.5 * cm, 5.5 * cm])
    bc_tbl.setStyle(_header_table_style(4))
    story.append(bc_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ==================================================================
    # 5. Correction Factors
    # ==================================================================
    story.append(Paragraph("  4. Correction Factors (IS 6403 Cl. 5.1.2)", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    cf_data = [
        ["Factor Type", "sc / dc / ic", "sq / dq / iq", "sγ / dγ / iγ", "Clause"],
        ["Shape  (s)", f"{results['s_c']:.4f}", f"{results['s_q']:.4f}",
         f"{results['s_gamma']:.4f}", "IS 6403 Table 2"],
        ["Depth  (d)", f"{results['d_c']:.4f}", f"{results['d_q']:.4f}",
         f"{results['d_gamma']:.4f}", "IS 6403 Cl. 5.1.2"],
        ["Incl.  (i)", f"{results['i_c']:.4f}", f"{results['i_q']:.4f}",
         f"{results['i_gamma']:.4f}", "IS 6403 Cl. 5.1.2"],
    ]
    cf_tbl = Table(cf_data, colWidths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 5 * cm])
    cf_tbl.setStyle(_header_table_style(5))
    story.append(cf_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ==================================================================
    # 6. Water Table Correction
    # ==================================================================
    story.append(Paragraph("  5. Water Table Correction (IS 6403 Cl. 5.1.3)", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    dw_val = inputs.get("water_table_depth", 999)
    wt_data = [
        ["Parameter", "Symbol", "Value"],
        ["Water table depth below GL",     "dw",       f"{dw_val:.2f} m" if dw_val < 999 else "No influence"],
        ["Effective overburden pressure",   "q",        f"{results['q_overburden']:.3f} kPa"],
        ["WT correction — surcharge term",  "W'_q",     f"{results['W_prime_q']:.4f}"],
        ["WT correction — unit wt term",    "W'_γ",     f"{results['W_prime_gamma']:.4f}"],
    ]
    wt_tbl = Table(wt_data, colWidths=[8 * cm, 2.5 * cm, 7 * cm])
    wt_tbl.setStyle(_header_table_style(3))
    story.append(wt_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ==================================================================
    # 7. Capacity Equation & Terms
    # ==================================================================
    story.append(Paragraph("  6. Net Ultimate Bearing Capacity Equation", s["section"]))
    story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph(
        "q_nu = c·Nc·sc·dc·ic  +  q·(Nq−1)·sq·dq·iq·W'_q  +  "
        "0.5·γ·B·Nγ·sγ·dγ·iγ·W'_γ\n"
        f"     = {results['term_c']:.3f}  +  {results['term_q']:.3f}  +  "
        f"{results['term_gamma']:.3f}",
        s["eq"],
    ))
    story.append(Spacer(1, 0.15 * cm))

    term_data = [
        ["Term", "Formula", "Value (kPa)"],
        ["Cohesion term",
         f"c·Nc·sc·dc·ic = {results['c_eff']:.3f}×{results['N_c']:.3f}×"
         f"{results['s_c']:.3f}×{results['d_c']:.3f}×{results['i_c']:.3f}",
         f"{results['term_c']:.3f}"],
        ["Surcharge term",
         f"q·(Nq−1)·sq·dq·iq·W'_q = {results['q_overburden']:.3f}×"
         f"({results['N_q']:.3f}−1)×{results['s_q']:.3f}×"
         f"{results['d_q']:.3f}×{results['i_q']:.3f}×{results['W_prime_q']:.3f}",
         f"{results['term_q']:.3f}"],
        ["Unit weight term",
         f"0.5·γ·B·Nγ·sγ·dγ·iγ·W'γ",
         f"{results['term_gamma']:.3f}"],
        ["Net Ultimate (q_nu)", "Sum of three terms", f"{results['q_nu']:.3f}"],
    ]
    term_tbl = Table(term_data, colWidths=[3.5 * cm, 10.5 * cm, 3.5 * cm])
    term_tbl.setStyle(_header_table_style(3))
    story.append(term_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ==================================================================
    # 8. Final Results Summary
    # ==================================================================
    story.append(Paragraph("  7. Safe Bearing Capacity Summary (IS 1904:1986)", s["section"]))
    story.append(Spacer(1, 0.2 * cm))

    res_data = [
        ["Gross Ultimate Bearing Capacity    q_f",  f"{results['q_f']:.2f} kPa"],
        ["Net Ultimate Bearing Capacity      q_nu", f"{results['q_nu']:.2f} kPa"],
        [f"Net Safe Bearing Capacity          q_ns  (FoS={results['fos']})",
         f"{results['q_ns']:.2f} kPa"],
        ["Gross Safe Bearing Capacity        q_s",  f"{results['q_s']:.2f} kPa"],
    ]
    res_tbl = Table(res_data, colWidths=[12 * cm, 5.5 * cm])
    res_tbl.setStyle(_result_box_style())
    story.append(res_tbl)
    story.append(Spacer(1, 0.5 * cm))

    # ==================================================================
    # 9. Footer
    # ==================================================================
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GREY_LINE))
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "This report is generated by the IS 6403:1981 Web Estimator for educational and "
        "preliminary design purposes only. All results must be verified by a qualified "
        "Geotechnical Engineer before use in construction. | IS 6403:1981 · IS 1904:1986",
        s["footer"],
    ))

    doc.build(story)
    return buffer.getvalue()
