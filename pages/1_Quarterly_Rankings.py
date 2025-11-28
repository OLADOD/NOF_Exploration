# app.py — v3
# -------------------------------------------------------------
# NHS Provider Metrics Dashboard
# Layout: KPIs (top) → Chart (left) + Ranks-across-metrics (right/sticky) → Table
# Chart: Plotly (robust), x=Providers, y=% Value, ordered by Rank asc
# Provider drop-down is filtered by Quarter→Domain→Metric (+Region)
# -------------------------------------------------------------

import io
import re
import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import html  # for safe HTML escaping of metric names
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# ===================== Page & Styles =========================
st.set_page_config(
    page_title="NOF Quarterly Rankings",   # or Monthly
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"       # <— add this
)

# Home owns this flag. Pages must not set a default.
st.session_state.setdefault("remember_filters", True)
REMEMBER = st.session_state.get("remember_filters", True)

# NEW: simple helper to read a remembered value when enabled
def remembered(key: str, default):
    if REMEMBER and key in st.session_state:
        return st.session_state[key]
    return default

# ---- Global page CSS (inline; keeps sidebar; pulls content up) ----

def use_ui_css():
    p = Path(__file__).parents[1] / "assets" / "ui.css"
    if p.exists():
        css = p.read_text(encoding="utf-8")
        # add a tiny cache-buster comment using mtime so browser applies updates
        css += f"\n/* mtime:{int(p.stat().st_mtime)} */"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

use_ui_css()

# ===================== Header ======================

st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      }
      .kpi-card {
        border: 1px solid var(--kpi-border, #e6e6e6);
        border-radius: 14px;
        padding: 14px 16px;
        background: var(--kpi-bg, #ffffff);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      .kpi-title { font-size: 0.85rem; color: var(--kpi-title, #555); margin-bottom: 4px; }
      .kpi-value { font-size: 1.4rem; font-weight: 600; }

      /* Metric mini-cards (right panel) */
      .metric-card {
        border: 1px solid var(--kpi-border, #e6e6e6);
        border-radius: 14px;
        padding: 10px 12px;
        background: var(--kpi-bg, #ffffff);
        margin-bottom: 10px;
      }
      .metric-title { font-size: 0.9rem; color: var(--kpi-title, #555); margin-bottom: 6px; }
      .metric-rank { font-size: 1.15rem; font-weight: 600; }
      .metric-sub { font-size: 0.8rem; color: var(--kpi-title, #666); }
      
      .muted { color: var(--kpi-title, #666); }

      /* Dark-mode friendly (auto via OS/browser) */
      @media (prefers-color-scheme: dark) {
        :root {
          --kpi-bg: #ffffff;
          --kpi-border: #D9D9D9;
          --kpi-title: #797979;
        }
      }
      
    /* -------- small vertical gap utilities -------- */
    .vgap-3 { height: 3px; }
    
    /* Summary panel sized to match the 420px chart height */
    .summary-card{
    border: 1px solid var(--kpi-border, #e6e6e6);
    border-radius: 14px;
    background: var(--kpi-bg, #fff);
    padding: 16px 18px;
    min-height: 420px;          /* match chart height */
    display: flex; flex-direction: column; justify-content: space-between;
    }
    .summary-card p{ margin: 0 0 12px 0; line-height: 1.55; font-size: 0.98rem; }
    .summary-title{ font-weight: 600; font-size: 1.05rem; margin-bottom: 8px; }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================== Constants =============================
COLS_ORIG = [
    "Quarter","Domain","Metric","Region",
    "Provider Code","Provider Name",
    "Numerator","Denominator","% Value","Rank",
    "Rank_Region","Region_Size",
    "Months Covered","Covered Months"
]

def underscore(x: str) -> str:
    x = x.replace("%","Percent")
    return re.sub(r"\s+","_",x)

COLS_US = [underscore(c) for c in COLS_ORIG]
RENAME_MAP = dict(zip(COLS_ORIG, COLS_US))

(QUARTER, DOMAIN, METRIC, REGION,
 PROV_CODE, PROV_NAME,
 NUM, DEN, PCT_STR, RANK,
 RANK_REGION, REGION_SIZE,
 MONTHS_COV, COVERED_MONTHS) = COLS_US

HIGHLIGHT_HEX = "#FAE100"      # NHS yellow
BAR_NEUTRAL_HEX = "#D5DAE1"    # light grey
DEFAULT_PROVIDER_CODE = "RWP"

DOMAIN_ORDER = {"A&E": 0, "Cancer": 1, "RTT": 2, "Diagnostic": 3}
RHS_PANEL_TITLE = "Metrics within domain"


# ===================== Helpers ===============================
def clean_numeric_str_to_float(x: str):
    if x is None: return float("nan")
    if not isinstance(x,str): x = str(x)
    cleaned = re.sub(r"[^\d\.\-]", "", x.replace(",", "."))
    return pd.to_numeric(cleaned, errors="coerce")

def clean_numeric_str_to_int(x: str):
    val = clean_numeric_str_to_float(x)
    return pd.to_numeric(val, errors="coerce")

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str)

    # Mandatory columns
    missing = [c for c in COLS_ORIG if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[COLS_ORIG].rename(columns=RENAME_MAP).copy()

    # Trim whitespace
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Numerator/Denominator may be decimals in file → keep as float internally
    def _to_float(x):
        return pd.to_numeric(re.sub(r"[^\d\.\-]", "", str(x).replace(",", ".")), errors="coerce")

    df[NUM] = df[NUM].map(_to_float)
    df[DEN] = df[DEN].map(_to_float)

    # Integers
    for c in [RANK, RANK_REGION, REGION_SIZE, MONTHS_COV]:
        df[c] = pd.to_numeric(df[c].map(_to_float), errors="coerce").astype("Int64")

    # Percent as float 0–100 (handles if file provided 0–1)
    pct = pd.to_numeric(df[PCT_STR].map(_to_float), errors="coerce")
    df["Percent"] = pct.where(pct > 1.0, pct * 100)

    # Ordered categorical quarter
    df[QUARTER] = pd.Categorical(df[QUARTER], categories=pd.unique(df[QUARTER]), ordered=True)
    return df


def format_percent_display(value: float, metric_name: str) -> str:
    if pd.isna(value): return "---"
    if str(metric_name).strip().lower() == "52+ weeks": return f"{value:.2f}%"
    return f"{value:.1f}%"

def format_whole_round(x) -> str:
    """Standard rounding (half up) to whole number with thousands separator; '—' if NaN."""
    if pd.isna(x):
        return "—"
    try:
        # Use Decimal for true half-up behaviour
        q = Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return f"{int(q):,}"
    except Exception:
        return "—"

def format_region_rank(rank_val, size_val) -> str:
    """e.g., '3 out of 9' or '—' if missing."""
    if pd.isna(rank_val) or pd.isna(size_val):
        return "—"
    return f"{int(rank_val)} out of {int(size_val)}"

def format_overall_rank_text(rank_val, nat_total) -> str:
    """Plain text: '12 out of 136' or '—' if missing."""
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)} out of {int(nat_total)}"

def format_overall_rank_html(rank_val, nat_total) -> str:
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)}<span class='muted'> out of {int(nat_total)}</span>"

def provider_options(df_filtered: pd.DataFrame) -> list:
    tmp = df_filtered[[PROV_CODE, PROV_NAME]].drop_duplicates().sort_values(PROV_CODE)
    return [f"{r[PROV_CODE]} — {r[PROV_NAME]}" for _, r in tmp.iterrows()]

def extract_code_from_label(label: str) -> str:
    return label.split("—")[0].strip() if isinstance(label, str) and "—" in label else label

def render_kpi_card(title: str, value: str):
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

def make_download_bytes(df: pd.DataFrame, as_excel: bool = False) -> bytes:
    if as_excel:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Filtered")
        return output.getvalue()
    return df.to_csv(index=False).encode("utf-8")

def build_chart_plotly(chart_df: pd.DataFrame, chart_title: str):
    dfp = chart_df.copy()

    # Auto-scale if file used 0–1 ratios (already handled in loader, but harmless here)
    if pd.notna(dfp["Percent"].max()) and dfp["Percent"].max() <= 1.0:
        dfp["Percent"] = dfp["Percent"] * 100

    # Pre-format display fields
    dfp["PercentLabel"] = dfp.apply(lambda r: format_percent_display(r["Percent"], r[METRIC]), axis=1)
    dfp["NumLabel"] = dfp[NUM].map(format_whole_round)
    dfp["DenLabel"] = dfp[DEN].map(format_whole_round)
    dfp["RegionRankLabel"] = dfp.apply(lambda r: format_region_rank(r[RANK_REGION], r[REGION_SIZE]), axis=1)

    colors = dfp["Is_Selected"].map({True: HIGHLIGHT_HEX, False: BAR_NEUTRAL_HEX})
    fig = px.bar(dfp, x=PROV_CODE, y="Percent", title=chart_title)
    fig.update_traces(
        marker_color=colors,
        customdata=dfp[[PROV_CODE, PROV_NAME, REGION, "NumLabel", "DenLabel", "PercentLabel", RANK, "RegionRankLabel"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Region: %{customdata[2]}<br>"
            "Numerator: %{customdata[3]}<br>"
            "Denominator: %{customdata[4]}<br>"
            "% Value: %{customdata[5]}<br>"
            "Rank: %{customdata[6]}<br>"
            "Region rank: %{customdata[7]}<extra></extra>"
        ),
    )
    fig.update_layout(
        template="simple_white",
        title=dict(x=0),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Providers",
        xaxis_showticklabels=False,
        yaxis_title=None,
        yaxis_range=[0, None],
        yaxis_ticksuffix="%",
        bargap=0.15,
    )
    return fig


def render_metric_rank_panel(
    df: pd.DataFrame,
    provider_code: str | None,
    selected_quarter: str,
    selected_domain: str,
    panel_title: str = RHS_PANEL_TITLE,
):
    """Right-hand 'Metrics within domain' panel rendered as progress cards."""
    if not provider_code:
        st.markdown(
            (
                '<div class="metrics-panel">'
                f'<div class="metrics-panel-title">{html.escape(panel_title)}</div>'
                '<div class="metric-item"><div class="metric-name">'
                'Select a provider to see ranks across all metrics.'
                '</div></div></div>'
            ),
            unsafe_allow_html=True,
        )
        return

    scope = df[(df[QUARTER] == selected_quarter) & (df[DOMAIN] == selected_domain)].copy()
    metrics_all = sorted(scope[METRIC].dropna().unique().tolist())

    cards: list[str] = []
    for m in metrics_all:
        r = scope[(scope[METRIC] == m) & (scope[PROV_CODE] == provider_code)]
        pct_val = 0.0

        if r.empty:
            rank_disp = "—"
            pct_disp  = "—"
            reg_rank  = "—"
        else:
            rr = r.iloc[0]
            rank_disp = "—" if pd.isna(rr[RANK]) else f"{int(rr[RANK])}"
            pct_disp  = format_percent_display(rr["Percent"], rr[METRIC])
            reg_rank  = format_region_rank(rr[RANK_REGION], rr[REGION_SIZE])
            if pd.notna(rr["Percent"]):
                pct_val = float(rr["Percent"])

        # clamp width 0–100 for safety
        width = max(0.0, min(100.0, pct_val))
        # National denominator for *this metric* in the current quarter+domain
        nat_n = scope[scope[METRIC] == m][PROV_CODE].nunique()
        nat_label = format_overall_rank_text(rr[RANK] if not r.empty else pd.NA, nat_n)

        reg_label = f"Regional Rank: {reg_rank}" if reg_rank != "—" else "Regional Rank: —"

        card_html = (
            f'<div class="progress-card">'
            f'  <div class="progress-head">'
            f'    <div class="progress-name">{html.escape(str(m))}</div>'
            f'    <div class="progress-percent">{pct_disp}</div>'
            f'  </div>'
            f'  <div class="progress-track"><div class="progress-fill" style="width:{width:.2f}%;"></div></div>'
            f'  <div class="progress-caption">'
            f'    <span>{nat_label}</span>'
            f'    <span class="muted">{html.escape(reg_label)}</span>'
            f'  </div>'
            f'</div>'
        )
        cards.append(card_html)


    panel_html = (
        '<div class="metrics-panel cards-plain">'
        f'<div class="metrics-panel-title">{html.escape(panel_title)}</div>'
        '<div class="progress-list">'
        + "".join(cards) +
        '</div></div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)


def info_banner(msg: str):
    st.markdown(
        f"<div class='info-row'><div class='info-banner'>{msg}</div></div>",
        unsafe_allow_html=True,
    )

def kpi_card(title: str, value_html: str, delta_html: str | None = None, delta_class: str = "neu"):
    show_delta = (delta_html is not None) and (str(delta_html).strip() not in {"—", ""})
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value_html}</div>
          {f"<div class='kpi-delta {delta_class}'>{delta_html}</div>" if show_delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

def provider_name_from_code(df_scope, code: str) -> str:
    if not code: return ""
    s = df_scope.loc[df_scope["Provider_Code"] == code, "Provider_Name"].dropna()
    return s.iloc[0] if len(s) else ""

# -----------------------------------------------

def round_half_up_to_int(x):
    if pd.isna(x): return None
    try:
        return int(Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except Exception:
        return None

def format_int_round(x) -> str:
    v = round_half_up_to_int(x)
    return "—" if v is None else f"{v:,}"

def weighted_percent(df_scope: pd.DataFrame) -> float | None:
    """Return 100 * sum(NUM) / sum(DEN) (NaN if not possible)."""
    num = pd.to_numeric(df_scope[NUM], errors="coerce").sum(min_count=1)
    den = pd.to_numeric(df_scope[DEN], errors="coerce").sum(min_count=1)
    if pd.isna(num) or pd.isna(den) or den == 0: return float("nan")
    return (num / den) * 100.0

def build_quarter_trend_df(df: pd.DataFrame,
                           provider_code: str | None,
                           selected_domain: str,
                           selected_metric: str,
                           region_selected: str | None) -> pd.DataFrame:
    """Returns per-quarter Percent for the current selection.
       Provider selected → that provider; else region weighted; else national weighted."""
    quarters = list(df[QUARTER].cat.categories)
    if provider_code:  # provider trend
        scope = df[(df[DOMAIN]==selected_domain) & (df[METRIC]==selected_metric) & (df[PROV_CODE]==provider_code)]
        if region_selected is not None:
            scope = scope[scope[REGION] == region_selected]
        agg = (scope
               .groupby(QUARTER, observed=True, sort=False)
               .agg(Percent=("Percent", "mean"),   # display Percent as recorded (already 0–100)
                    Numerator=(NUM, "sum"), Denominator=(DEN, "sum"))
               .reindex(quarters))
    elif region_selected is not None:  # region weighted
        scope = df[(df[DOMAIN]==selected_domain) & (df[METRIC]==selected_metric) & (df[REGION]==region_selected)]
        rows = []
        for q in quarters:
            qdf = scope[scope[QUARTER]==q]
            rows.append((q, weighted_percent(qdf)))
        agg = pd.DataFrame(rows, columns=[QUARTER, "Percent"]).set_index(QUARTER)
    else:  # national weighted
        scope = df[(df[DOMAIN]==selected_domain) & (df[METRIC]==selected_metric)]
        rows = []
        for q in quarters:
            qdf = scope[scope[QUARTER]==q]
            rows.append((q, weighted_percent(qdf)))
        agg = pd.DataFrame(rows, columns=[QUARTER, "Percent"]).set_index(QUARTER)

    agg = agg.reset_index()
    # Display label with correct 1dp / 2dp rule
    agg["PercentLabel"] = [format_percent_display(p, selected_metric) for p in agg["Percent"]]
    return agg

def make_summary_html(df: pd.DataFrame,
                      df_all: pd.DataFrame,
                      provider_code: str | None,
                      quarter: str,
                      domain: str,
                      metric: str,
                      region_selected: str | None) -> str:
    """Return narrative HTML summary matching the current selection."""
    
    # Common counts
    nat_scope_now = df_all[(df_all[QUARTER]==quarter) & (df_all[DOMAIN]==domain) & (df_all[METRIC]==metric)]
    nat_n = nat_scope_now[PROV_CODE].nunique()

    if provider_code:
        row = df[(df[PROV_CODE]==provider_code)].iloc[0]
        prov_name = provider_name_from_code(df, provider_code) or provider_code
        
        # Format current values
        pct_disp = format_percent_display(row["Percent"], row[METRIC])
        num_disp = format_int_round(row[NUM])
        den_disp = format_int_round(row[DEN])
        rank_val = int(row[RANK]) if pd.notna(row[RANK]) else None
        reg_rank_val = int(row[RANK_REGION]) if pd.notna(row[RANK_REGION]) else None
        reg_size_val = int(row[REGION_SIZE]) if pd.notna(row[REGION_SIZE]) else None
        cov_disp = row[COVERED_MONTHS].strip() if isinstance(row[COVERED_MONTHS], str) and row[COVERED_MONTHS].strip() else "Jul, Aug, Sep"
        
        provider_region = (row[REGION] if isinstance(row.get(REGION, None), (str,)) and str(row[REGION]).strip()
                           else (region_selected or ""))

        # Calculate previous quarter changes
        quarters = list(df_all[QUARTER].cat.categories)
        try:
            i = quarters.index(quarter)
            prev_q = quarters[i-1] if i > 0 else None
        except ValueError:
            prev_q = None

        perf_change_txt = ""
        rank_change_txt = ""
        
        if prev_q:
            prev = df_all[(df_all[PROV_CODE]==provider_code) & (df_all[DOMAIN]==domain) & 
                         (df_all[METRIC]==metric) & (df_all[QUARTER]==prev_q)]
            if not prev.empty:
                p = prev.iloc[0]
                # Performance delta
                if pd.notna(row["Percent"]) and pd.notna(p["Percent"]):
                    dpp = row["Percent"] - p["Percent"]
                    if abs(dpp) >= 0.05:  # Only show if meaningful change
                        direction = "improved" if dpp > 0 else "declined"
                        perf_change_txt = f"Performance {direction} by {abs(dpp):.1f} percentage points compared to {prev_q}"
                    else:
                        perf_change_txt = f"Performance remained stable compared to {prev_q}"
                
                # Rank delta
                if pd.notna(row[RANK]) and pd.notna(p[RANK]):
                    dr = int(p[RANK]) - int(row[RANK])  # positive = improved
                    if dr > 0:
                        rank_change_txt = f", climbing {abs(dr)} position{'s' if abs(dr) > 1 else ''} in the national rankings"
                    elif dr < 0:
                        rank_change_txt = f", dropping {abs(dr)} position{'s' if abs(dr) > 1 else ''} in the national rankings"

        # Calculate regional and national weighted averages for comparison
        reg_scope = nat_scope_now[nat_scope_now[REGION] == provider_region] if provider_region else pd.DataFrame()
        reg_avg = weighted_percent(reg_scope) if not reg_scope.empty else None
        nat_avg = weighted_percent(nat_scope_now)
        
        # Comparison context
        comparison_txt = ""
        if pd.notna(row["Percent"]):
            comparisons = []
            if reg_avg and pd.notna(reg_avg):
                diff_reg = row["Percent"] - reg_avg
                if abs(diff_reg) >= 0.5:
                    comp = "above" if diff_reg > 0 else "below"
                    comparisons.append(f"{abs(diff_reg):.1f}pp {comp} the regional average of {format_percent_display(reg_avg, metric)}")
            
            if nat_avg and pd.notna(nat_avg):
                diff_nat = row["Percent"] - nat_avg
                if abs(diff_nat) >= 0.5:
                    comp = "above" if diff_nat > 0 else "below"
                    comparisons.append(f"{abs(diff_nat):.1f}pp {comp} the national average of {format_percent_display(nat_avg, metric)}")
            
            if comparisons:
                comparison_txt = f" This is {' and '.join(comparisons)}."

        # Build narrative paragraphs
        p1 = (
            f"<p><b>{html.escape(prov_name)}</b> achieved a <b>{pct_disp}</b> performance rate for "
            f"<b>{html.escape(domain)} → {html.escape(metric)}</b> in <b>{html.escape(quarter)}</b> across the <b>{html.escape(provider_region)}</b> region. "
            f"This placed them <b>{rank_val}{'th' if rank_val else ''} nationally</b> out of {nat_n} trusts"
        )
        
        if reg_rank_val and reg_size_val:
            p1 += f", and <b>{reg_rank_val}{'th' if reg_rank_val else ''} within their region</b> of {reg_size_val} trusts"
        
        p1 += f".{comparison_txt}</p>"

        # Second paragraph: volume and data coverage
        p2 = (
            f"<p>The trust reported <b>{num_disp}</b> cases meeting the standard "
            f"out of <b>{den_disp}</b> total cases during this quarter"
        )
        
        if cov_disp and cov_disp != "—":
            p2 += f", covering <b>{cov_disp}</b>"
        
        p2 += "."
        
        # Add change narrative if available
        if perf_change_txt:
            p2 += f" {perf_change_txt}{rank_change_txt}."
        
        p2 += "</p>"

        # Third paragraph: chart guidance
        p3 = (
            "<p>The bar chart displays all trusts ranked by performance, with your selected trust "
            "highlighted in yellow for easy identification. The trend chart below tracks quarterly "
            "performance over time, showing the trust's trajectory (solid blue line) alongside "
            "regional (dashed grey) and national (dotted teal) weighted averages for comparison.</p>"
        )

        return (
            "<div class='metrics-panel summary-panel'>"
            f"<div class='summary-title'>Summary</div>"
            f"{p1}{p2}{p3}"
            "</div>"
        )

    # Region summary (no provider selected)
    if region_selected:
        scope_now = nat_scope_now[nat_scope_now[REGION]==region_selected]
        reg_n = scope_now[PROV_CODE].nunique()
        reg_pct = weighted_percent(scope_now)
        reg_pct_disp = format_percent_display(reg_pct, metric)
        
        # National average for comparison
        nat_avg = weighted_percent(nat_scope_now)
        
        comparison_txt = ""
        if reg_pct and nat_avg and pd.notna(reg_pct) and pd.notna(nat_avg):
            diff = reg_pct - nat_avg
            if abs(diff) >= 0.5:
                comp = "above" if diff > 0 else "below"
                comparison_txt = f" This is {abs(diff):.1f} percentage points {comp} the national weighted average of {format_percent_display(nat_avg, metric)}."
        
        # Previous quarter comparison
        quarters = list(df_all[QUARTER].cat.categories)
        try:
            i = quarters.index(quarter)
            prev_q = quarters[i-1] if i > 0 else None
        except ValueError:
            prev_q = None
        
        change_txt = ""
        if prev_q:
            prev_scope = df_all[(df_all[DOMAIN]==domain) & (df_all[METRIC]==metric) & 
                               (df_all[REGION]==region_selected) & (df_all[QUARTER]==prev_q)]
            prev_pct = weighted_percent(prev_scope)
            if reg_pct and prev_pct and pd.notna(reg_pct) and pd.notna(prev_pct):
                dpp = reg_pct - prev_pct
                if abs(dpp) >= 0.05:
                    direction = "improved" if dpp > 0 else "declined"
                    change_txt = f" Regional performance {direction} by {abs(dpp):.1f} percentage points compared to {prev_q}."

        p1 = (
            f"<p>The <b>{html.escape(region_selected)}</b> region achieved a weighted average performance of "
            f"<b>{reg_pct_disp}</b> for <b>{html.escape(domain)} → {html.escape(metric)}</b> in <b>{html.escape(quarter)}</b>. "
            f"This regional view includes <b>{reg_n} trusts</b> out of <b>{nat_n}</b> trusts nationally."
            f"{comparison_txt}{change_txt}</p>"
        )
        
        p2 = (
            "<p>The trend chart displays quarterly performance for the region (dashed grey line) "
            "compared to the national weighted average (dotted teal line). Select a specific provider "
            "from the dropdown above to see individual trust performance, rankings, and detailed KPIs.</p>"
        )
        
        return (
            "<div class='metrics-panel summary-panel'>"
            f"<div class='summary-title'>Summary</div>"
            f"{p1}{p2}"
            "</div>"
        )

    # National summary (no provider or region selected)
    nat_pct = weighted_percent(nat_scope_now)
    nat_pct_disp = format_percent_display(nat_pct, metric)
    
    # Previous quarter comparison
    quarters = list(df_all[QUARTER].cat.categories)
    try:
        i = quarters.index(quarter)
        prev_q = quarters[i-1] if i > 0 else None
    except ValueError:
        prev_q = None
    
    change_txt = ""
    if prev_q:
        prev_scope = df_all[(df_all[DOMAIN]==domain) & (df_all[METRIC]==metric) & (df_all[QUARTER]==prev_q)]
        prev_pct = weighted_percent(prev_scope)
        if nat_pct and prev_pct and pd.notna(nat_pct) and pd.notna(prev_pct):
            dpp = nat_pct - prev_pct
            if abs(dpp) >= 0.05:
                direction = "improved" if dpp > 0 else "declined"
                change_txt = f" National performance {direction} by {abs(dpp):.1f} percentage points compared to {prev_q}."
            else:
                change_txt = f" National performance remained stable compared to {prev_q}."

    p1 = (
        f"<p>Across all regions in <b>{quarter}</b>, the national weighted average for "
        f"<b>{domain} → {metric}</b> was <b>{nat_pct_disp}</b>, based on data from "
        f"<b>{nat_n} trusts</b>.{change_txt}</p>"
    )
    
    p2 = (
        "<p>The trend chart shows quarterly performance at the national level (dotted teal line). "
        "To explore regional variations or individual trust performance, select a region or provider "
        "from the filters above. This will reveal detailed rankings, KPIs, and comparative insights.</p>"
    )
    
    return (
        "<div class='metrics-panel summary-panel'>"
        f"<div class='summary-title'>Summary</div>"
        f"{p1}{p2}"
        "</div>"
    )

def resolve_region_for_compare(df: pd.DataFrame,
                               domain: str,
                               metric: str,
                               provider_code: str | None,
                               region_selected: str | None) -> str | None:
    """Pick which region to use for the dashed comparison line.
       Priority: explicit Region filter → provider's most-recent Region → None."""
    if region_selected:
        return region_selected
    if provider_code:
        scope = df[(df[DOMAIN] == domain) & (df[METRIC] == metric) & (df[PROV_CODE] == provider_code)]
        if not scope.empty:
            # QUARTER is an ordered categorical; sorting respects data order
            latest_row = scope.sort_values(QUARTER).iloc[-1]
            val = str(latest_row[REGION]).strip()
            return val if val else None
    return None

def build_quarter_trend_lines_df(df: pd.DataFrame,
                                 domain: str,
                                 metric: str,
                                 provider_code: str | None,
                                 region_for_compare: str | None) -> pd.DataFrame:
    """Return a dataframe with Provider, RegionWeighted, NationalWeighted by Quarter,
       plus pre-formatted labels with your 1dp/2dp rule."""
    quarters = list(df[QUARTER].cat.categories)
    rows = []
    for q in quarters:
        qscope = df[(df[DOMAIN] == domain) & (df[METRIC] == metric) & (df[QUARTER] == q)]

        # Provider % (as recorded 0–100)
        if provider_code:
            s = qscope.loc[qscope[PROV_CODE] == provider_code, "Percent"]
            prov = float(s.iloc[0]) if len(s) else float("nan")
        else:
            prov = float("nan")

        # Region weighted % from counts
        reg = weighted_percent(qscope[qscope[REGION] == region_for_compare]) if region_for_compare else float("nan")

        # National weighted % from counts
        nat = weighted_percent(qscope)

        rows.append({QUARTER: q, "Provider": prov, "RegionWeighted": reg, "NationalWeighted": nat})

    trend = pd.DataFrame(rows)
    # Pre-format labels with your display rule
    trend["ProvLbl"] = [format_percent_display(v, metric) for v in trend["Provider"]]
    trend["RegLbl"]  = [format_percent_display(v, metric) for v in trend["RegionWeighted"]]
    trend["NatLbl"]  = [format_percent_display(v, metric) for v in trend["NationalWeighted"]]
    return trend

# ===== Empty state (when no CSV) =====
def render_empty_state(page_name: str, template_cols: list[str], demo_rel_path: str | None = None):
    # Top banner (kept)
    st.markdown(
        f'<div id="context-banner">👋 Upload a CSV in the sidebar to begin.</div>',
        unsafe_allow_html=True
    )

    # Use the SAME padded wrapper as the banner so left/right edges align
    st.markdown("<div class='info-row empty-grid'>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")

    # --- Left: quick start & template download ---
    with left:
        st.markdown("### Get started")
        st.markdown(
            "1. **Download the template** (headers only).  \n"
            "2. **Fill the required columns** in your CSV.  \n"
            "3. **Upload via the sidebar.**"
        )

        tpl = pd.DataFrame(columns=template_cols)
        buf = io.BytesIO()
        tpl.to_csv(buf, index=False)
        st.download_button("⬇️ Download template CSV", buf.getvalue(),
                           file_name=f"{page_name.lower()}_template.csv",
                           use_container_width=True)

        with st.expander("❓ FAQ & troubleshooting", expanded=False):
            st.markdown(
                "- **Missing columns?** File must include all headers shown on the right.  \n"
                "- **% Value scale:** If your file uses 0–1, we auto-scale ×100.  \n"
                "- **Large files:** Up to 200 MB. Remove blanks; keep only needed columns.  \n"
                "- **Privacy:** Files are processed locally in this session."
            )

    # --- Right: the rounded “metrics panel” with two headings (no tabs) ---
    with right:
        panel_html = """
        <div class="metrics-panel required-cols">
        <div class="metrics-panel-title">Required columns</div>
        <div class="panel-subtitle">Quarterly</div>
        <div class="schema"><code>Quarter, Domain, Metric, Region, Provider Code, Provider Name, Numerator, Denominator, % Value, Rank, Rank_Region, Region_Size, Months Covered, Covered Months</code></div>
        <div class="panel-subtitle">Monthly</div>
        <div class="schema"><code>Month, Domain, Metric, Region, Provider Code, Provider Name, Numerator, Denominator, % Value, Rank, Rank_Region, Region_Size, Data_Date_Used</code></div>
        </div>
        """
        st.markdown(panel_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close .info-row.empty-grid

def format_overall_rank(rank_val, nat_total) -> str:
    """e.g., '12 out of 136' or '—' if missing."""
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)}<span class='muted'> out of {int(nat_total)}</span>"

# ------------------------ End of def -----------------------------


# ===================== Sidebar (Upload + Filters) =============
with st.sidebar:
    st.header("📁 Quarterly data")

    has_quarterly = ("quarterly_bytes" in st.session_state)

    if has_quarterly:
        st.caption(f"Using: {st.session_state.get('quarterly_name', '(uploaded)')}")
        if st.button("Clear file", key="clear_quarterly"):
            st.session_state.pop("quarterly_bytes", None)
            st.session_state.pop("quarterly_name",  None)
            st.rerun()
    else:
        st.warning("No Quarterly CSV loaded. Go to the Homepage to upload.", icon="⚠️")
        if st.button("Go to Homepage", key="goto_home_for_quarterly"):
            st.switch_page("Homepage.py")
    # ← No st.stop() here


# If no file yet → show empty state and stop (Monthly)
if "quarterly_bytes" not in st.session_state:
    render_empty_state(
        page_name="Quarterly",
        template_cols=[ "Quarter","Domain","Metric","Region",
                        "Provider Code","Provider Name","Numerator","Denominator",
                        "% Value","Rank","Rank_Region","Region_Size",
                        "Months Covered","Covered Months" ],
        demo_rel_path=None
    )
    st.stop()


# ===================== Load Data ===============================
# load_csv already expects a file-like object; wrap the bytes
try:
    import io  # (already imported at top in your file; harmless to re-import)
    df = load_csv(io.BytesIO(st.session_state["quarterly_bytes"]))
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# 1) Quarter
quarter_options = list(df[QUARTER].cat.categories)
default_quarter = quarter_options[-1] if quarter_options else None
q_quarter = remembered("q_quarter", default_quarter)
if q_quarter not in quarter_options:
    q_quarter = default_quarter or (quarter_options[0] if quarter_options else None)
quarter = st.sidebar.selectbox("Quarter", quarter_options,
                               index=quarter_options.index(q_quarter) if q_quarter else 0)

df_q = df[df[QUARTER] == quarter]

# 2) Domain (custom order instead of alphabetical)
domain_options = sorted(df_q[DOMAIN].dropna().unique().tolist(),
                        key=lambda d: DOMAIN_ORDER.get(d, 999))
q_domain = remembered("q_domain", domain_options[0] if domain_options else "")
if q_domain not in domain_options and domain_options:
    q_domain = domain_options[0]
domain = st.sidebar.selectbox("Domain", domain_options,
                              index=domain_options.index(q_domain) if domain_options else 0)

df_qd = df_q[df_q[DOMAIN] == domain]

# 3) Metric (depends on Domain)
metric_options = sorted(df_qd[METRIC].dropna().unique().tolist())
q_metric = remembered("q_metric", metric_options[0] if metric_options else "")
if q_metric not in metric_options and metric_options:
    q_metric = metric_options[0]
metric = st.sidebar.selectbox("Metric", metric_options,
                              index=metric_options.index(q_metric) if metric_options else 0)

df_qdm = df_qd[df_qd[METRIC] == metric]

# 4) Region (optional; no selection = all)
region_options = ["(All Regions)"] + sorted(df_qdm[REGION].dropna().unique().tolist())
q_region = remembered("q_region", "(All Regions)")
if q_region not in region_options:
    q_region = "(All Regions)"
region_choice = st.sidebar.selectbox("Region", region_options,
                                     index=region_options.index(q_region))
region_selected = None if region_choice == "(All Regions)" else region_choice
df_qdmr = df_qdm if region_selected is None else df_qdm[df_qdm[REGION] == region_selected]

# 5) Provider (optional; filtered by Quarter+Domain+Metric and Region if set)
prov_opts_labels = provider_options(df_qdmr)
default_provider_label = next(
    (lbl for lbl in prov_opts_labels if extract_code_from_label(lbl) == DEFAULT_PROVIDER_CODE),
    None
)

# Shared provider from other pages (kept)
shared_provider = st.session_state.get("shared_provider_code", None) if REMEMBER else None
# NEW: read remembered local label safely
provider_label_rem = st.session_state.get("q_provider", "(None)")

# Determine the default index with clear precedence:
# 1) shared provider (if available in current options)
# 2) remembered local provider label
# 3) hard-coded default provider
# 4) "(None)"
if shared_provider and any(extract_code_from_label(lbl) == shared_provider for lbl in prov_opts_labels):
    matching_label = next((lbl for lbl in prov_opts_labels if extract_code_from_label(lbl) == shared_provider), None)
    default_index = (["(None)"] + prov_opts_labels).index(matching_label) if matching_label else 0
elif provider_label_rem in (["(None)"] + prov_opts_labels):
    default_index = (["(None)"] + prov_opts_labels).index(provider_label_rem)
elif default_provider_label:
    default_index = prov_opts_labels.index(default_provider_label) + 1
else:
    default_index = 0

provider_label = st.sidebar.selectbox(
    "Provider (optional)",
    options=["(None)"] + prov_opts_labels,
    index=default_index,
    help="Selecting a provider highlights it and shows KPIs."
)
provider_code = None if provider_label == "(None)" else extract_code_from_label(provider_label)

# Save current Quarterly filters to session (so they survive page hops)
if REMEMBER:
    st.session_state.update({
        "q_quarter": quarter,
        "q_domain": domain,
        "q_metric": metric,
        "q_region": region_choice,
        "q_provider": provider_label,
    })

# Keep existing cross-page behaviour: only update shared provider if a real one is chosen
if REMEMBER and provider_code:
    st.session_state["shared_provider_code"] = provider_code


# Always save the provider code to shared state (for cross-page sync)
# Only update shared when a real provider is selected AND remember is ON
if REMEMBER and provider_code:
    st.session_state["shared_provider_code"] = provider_code


ROOT_ASSETS  = Path(__file__).parents[1] / "assets"           # project root /assets
PAGES_ASSETS = Path(__file__).parent / "assets"               # pages/assets (fallback)
for candidate in (ROOT_ASSETS / "NOF_Logo.svg", PAGES_ASSETS / "NOF_Logo.svg"):
    if candidate.is_file():
        LOGO_FILE = candidate
        break
else:
    LOGO_FILE = None

def read_svg_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(f"Logo not found at {p}. Rendering title without logo.")
        return ""
    except Exception as e:
        st.warning(f"Could not read logo ({e}).")
        return ""

logo_svg = read_svg_file(LOGO_FILE) if LOGO_FILE else ""

# ===================== Header ======================

st.markdown(
    f"""
    <div class="app-header" role="banner" aria-label="Header">
      <div class="app-logo" aria-hidden="true">{logo_svg}</div>
      <h1 id="page-title">NOF Quarterly Rankings</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== Context banner ======================
context_html = (
    f'Showing <b>{html.escape(domain)}</b> → <b>{html.escape(metric)}</b> in <b>{html.escape(quarter)}</b>'
    + (f' for <b>{html.escape(region_selected)}</b> region' if region_selected else ' across <b>all regions</b>')
    + '.'
)
st.markdown(f'<div id="context-banner">{context_html}</div>', unsafe_allow_html=True)

# ===================== Provider heading ====================
if provider_code:
    prov_name = provider_name_from_code(df_qdmr, provider_code)
    st.markdown(f"<h2 class='kpi-heading'>{html.escape(prov_name)} ({html.escape(provider_code)})</h2>", unsafe_allow_html=True)
else:
    st.info("Select a provider to see KPIs and trend.")

# ===================== KPI cards (no deltas on Quarterly) ===
if provider_code:
    row = df_qdmr.loc[df_qdmr[PROV_CODE] == provider_code]
    if not row.empty:
        r = row.iloc[0]
        # (A) Overall rank as "X out of Y" using the helper
        # ---- Overall rank as "X out of Y" (plain text) ----
        nat_n = df_qdm[PROV_CODE].nunique()  # national size for this Quarter+Domain+Metric
        overall_text = format_overall_rank_html(r[RANK], nat_n)

        # ---- Region rank "X out of Y" (plain text) ----
        region_rank_text = (
            "—" if pd.isna(r[RANK_REGION]) or pd.isna(r[REGION_SIZE])
            else f"{int(r[RANK_REGION])} out of {int(r[REGION_SIZE])}"
        )

        num_disp = format_whole_round(r[NUM])
        den_disp = format_whole_round(r[DEN])
        pct_disp = format_percent_display(r['Percent'], r[METRIC])
        cov_disp = r[COVERED_MONTHS].strip() if isinstance(r[COVERED_MONTHS], str) and r[COVERED_MONTHS].strip() else "—"

        # Order: Rank → Region Rank → Numerator → Denominator → % Value → Covered_Months
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi_card("Overall Rank", overall_text)        # ← title AND new value
        with c2: kpi_card("Region Rank",  region_rank_text)
        with c3: kpi_card("Numerator",    num_disp)
        with c4: kpi_card("Denominator",  den_disp)
        with c5: kpi_card("% Value",      pct_disp)
        with c6: kpi_card("Covered_Months", cov_disp)

    else:
        st.warning("No data for the selected provider under the current Metric/Region.")

# small spacer before the chart + RHS panel
st.markdown('<div class="vgap-3"></div>', unsafe_allow_html=True)

# ===================== Distribution Chart ===============================

# Chart data (Region applied; Provider does NOT filter the set)
chart_df = df_qdmr.copy()
chart_df_plot = chart_df.dropna(subset=["Percent", RANK], how="any").copy()
chart_df_plot = chart_df_plot.sort_values([RANK, "Percent", PROV_NAME], ascending=[True, False, True])
chart_df_plot["Is_Selected"] = (chart_df_plot[PROV_CODE].eq(provider_code) if provider_code else False)
chart_df_plot["PercentLabel"] = chart_df_plot.apply(lambda r: format_percent_display(r["Percent"], r[METRIC]), axis=1)

if chart_df_plot.empty:
    total = len(chart_df); missing_pct = chart_df["Percent"].isna().sum(); missing_rank = chart_df[RANK].isna().sum()
    st.warning(f"No bars to draw. Rows: {total:,} · missing Percent: {missing_pct:,} · missing Rank: {missing_rank:,}")

chart_title = "Provider Performance (% Value) — ordered by Rank (1 at left)"
fig = build_chart_plotly(chart_df_plot, chart_title)
# Remove the bottom x-axis line (and any grid/zero line)
fig.update_xaxes(
    showline=False,     # turn off the axis baseline
    linewidth=0,        # belt-and-braces
    linecolor="rgba(0,0,0,0)",
    showgrid=False,
    zeroline=False,
    ticks="",           # no tick marks
    mirror=False
)
TREND_H = 450
    
fig.update_layout(height=TREND_H)
st.plotly_chart(fig, use_container_width=True)
# Add a thin separator line above the chart
st.markdown('<hr class="chart-separator">', unsafe_allow_html=True)

# -------- Trend + Summary row (keeps your earlier logic) --------
st.markdown('<div class="vgap-3"></div>', unsafe_allow_html=True)
t_left, t_right = st.columns([0.60, 0.40], gap="medium")

with t_left:
    region_for_compare = resolve_region_for_compare(df, domain, metric, provider_code, region_selected)
    trend = build_quarter_trend_lines_df(df, domain, metric, provider_code, region_for_compare)

    fig_trend = go.Figure()
    trace_kwargs = dict(mode="lines+markers", cliponaxis=False)

    # Provider — solid blue, thicker
    if trend["Provider"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend[QUARTER], y=trend["Provider"],
            name=(provider_code or "Provider"),
            line=dict(width=3, color="#327AD1"),   # solid blue
            customdata=trend["ProvLbl"],
            hovertemplate = "<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))

    # Region (weighted) — dashed grey
    if region_for_compare and trend["RegionWeighted"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend[QUARTER], y=trend["RegionWeighted"],
            name=f"{region_for_compare} (weighted)",
            line=dict(width=2, dash="dash", color="#6B7280"),
            customdata=trend["RegLbl"],
            hovertemplate = "<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))

    # National (weighted) — dotted teal
    if trend["NationalWeighted"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend[QUARTER], y=trend["NationalWeighted"],
            name="National (weighted)",
            line=dict(width=2, dash="dot", color="#0C988F"),
            customdata=trend["NatLbl"],
            hovertemplate = "<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))
    
    fig_trend.update_layout(
        title=dict(text=f"{metric} — Quarterly Trend", x=0, xanchor="left"),
        template="simple_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.10, yanchor="top", x=-0.05, xanchor="left"),
        height=320,                               # keep in sync with summary min-height if you set one
        margin=dict(l=10, r=10, t=50, b=6),
        hoverlabel=dict(namelength=-1),
    )
    fig_trend.update_yaxes(
        showgrid=True, 
        gridcolor="#E5E7EB", 
        gridwidth=1, 
        ticksuffix="%",
        autorange=True, 
        automargin=True,
        showline=False,            # HIDE the bottom axis baseline
    )
    
    fig_trend.update_xaxes(
        title_text=None,
        showgrid=False,            # no vertical gridlines
        showline=False,            # HIDE the bottom axis baseline
        zeroline=False,            # no extra line at x=0
        linewidth=0,
        linecolor="rgba(0,0,0,0)",
        ticks="",                  # no tick marks
        mirror=False
    )

    
    TREND_H = 450
    fig_trend.update_layout(autosize=False, height=TREND_H)

    st.plotly_chart(fig_trend, use_container_width=True)



with t_right:
    st.markdown('<div class="rhs-sticky">', unsafe_allow_html=True)
    render_metric_rank_panel(df, provider_code, quarter, domain, panel_title=RHS_PANEL_TITLE)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("<div class='summary-offset'>", unsafe_allow_html=True)
html_summary = make_summary_html(
    df_qdmr if provider_code else df,
    df, provider_code, quarter, domain, metric, region_selected
)
st.markdown(html_summary, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="vgap-3"></div>', unsafe_allow_html=True)

# ===================== Table (full width) =====================
with st.expander("See filtered data as a table and download"):
    table_cols = [
        QUARTER, DOMAIN, METRIC, REGION,
        PROV_CODE, PROV_NAME,
        NUM, DEN, PCT_STR, RANK,
        RANK_REGION, REGION_SIZE,
        MONTHS_COV, COVERED_MONTHS
    ]
    table_df = df_qdmr[table_cols].copy()

    # Display formats
    table_df[NUM] = table_df[NUM].map(format_whole_round)
    table_df[DEN] = table_df[DEN].map(format_whole_round)
    table_df[PCT_STR] = [format_percent_display(p, m) for p, m in zip(df_qdmr["Percent"], df_qdmr[METRIC])]
    # RANK/RANK_REGION/REGION_SIZE are Int64; display as plain ints/— automatically in dataframe

    st.dataframe(table_df, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download CSV",
            data=make_download_bytes(table_df, False),
            file_name="filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download Excel",
            data=make_download_bytes(table_df, True),
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ===================== Notes ================================
st.markdown(
    """
    <div id="footer-notes" style="
        color:#454545;
        opacity:1 !important;
        font-size:0.92rem;
        line-height:1.7;
        margin-top:6px;
    ">
      <ul style="margin:0; padding-left:1.2rem;">
        <li>Column headers are normalized internally (e.g., <code style='color:#1AA855;background:rgba(144,204,169,0.26);border-radius:4px;padding:0 .25rem;'>Provider_Code</code>).</li>
        <li>Bars are ordered by <b>Rank</b> (ascending).</li>
        <li><code style='color:#327AD1;background:rgba(193,221,255,.25);border-radius:4px;padding:0 .25rem;'>% Value</code> uses 2dp only for <i>52+ Weeks</i>; others 1dp.</li>
        <li>Missing values are hidden in charts and shown as <code style='color:#F04141;background:rgba(255,239,193,.85);border-radius:4px;padding:0 .25rem;'>---</code> in KPIs/table.</li>
        <li>Right-hand panel shows the selected provider’s <b>Rank across all metrics</b> in the chosen <b>Quarter + Domain</b>.</li>
        <p><i>Developed by: David M. Oladoyin</i></p>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)