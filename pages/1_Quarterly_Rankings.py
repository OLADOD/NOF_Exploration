# app.py — v3
# -------------------------------------------------------------
# NHS Provider Metrics Dashboard
# Layout: KPIs (top) → Chart (left) + Ranks-across-metrics (right/sticky) → Table
# Chart: Plotly (robust), x=Providers, y=% Value, ordered by Rank asc
# Provider drop-down is filtered by Quarter→Domain→Metric (+Region)
# -------------------------------------------------------------

import io
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import html  # for safe HTML escaping of metric names
from pathlib import Path


# ===================== Page & Styles =========================
st.set_page_config(
    page_title="NOF Quarterly Rankings",   # or Monthly
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"       # <— add this
)

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
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ===================== Constants =============================
COLS_ORIG = [
    "Quarter","Domain","Metric","Region",
    "Provider Code","Provider Name",
    "Numerator","Denominator","% Value","Rank",
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
 MONTHS_COV, COVERED_MONTHS) = COLS_US

HIGHLIGHT_HEX = "#FAE100"      # NHS yellow
BAR_NEUTRAL_HEX = "#D5DAE1"    # light grey
DEFAULT_PROVIDER_CODE = "RWP"

# Domain order for the Domain select (your custom order)
DOMAIN_ORDER = {"A&E": 0, "Cancer": 1, "RTT": 2, "Diagnostic": 3}

RHS_PANEL_TITLE = "📋Metrics within domain"  # change anytime

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
    missing = [c for c in COLS_ORIG if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[COLS_ORIG].rename(columns=RENAME_MAP).copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    for c in [NUM, DEN, RANK, MONTHS_COV]:
        df[c] = df[c].apply(clean_numeric_str_to_int)
    df["Percent"] = df[PCT_STR].apply(clean_numeric_str_to_float)
    df[QUARTER] = pd.Categorical(df[QUARTER], categories=pd.unique(df[QUARTER]), ordered=True)
    return df

def format_percent_display(value: float, metric_name: str) -> str:
    if pd.isna(value): return "---"
    if str(metric_name).strip().lower() == "52+ weeks": return f"{value:.2f}%"
    return f"{value:.1f}%"

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
    # Auto-scale if file used 0–1 ratios
    if pd.notna(chart_df["Percent"].max()) and chart_df["Percent"].max() <= 1.0:
        chart_df = chart_df.copy()
        chart_df["Percent"] = chart_df["Percent"] * 100
        chart_df["PercentLabel"] = chart_df.apply(lambda r: format_percent_display(r["Percent"], r[METRIC]), axis=1)

    colors = chart_df["Is_Selected"].map({True: HIGHLIGHT_HEX, False: BAR_NEUTRAL_HEX})
    fig = px.bar(chart_df, x=PROV_CODE, y="Percent", title=chart_title)
    fig.update_traces(
        marker_color=colors,
        customdata=chart_df[[PROV_CODE, PROV_NAME, REGION, NUM, DEN, "PercentLabel", RANK]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Region: %{customdata[2]}<br>"
            "Numerator: %{customdata[3]:,}<br>"
            "Denominator: %{customdata[4]:,}<br>"
            "% Value: %{customdata[5]}<br>"
            "Rank: %{customdata[6]:,}<extra></extra>"
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
    """Right-hand sticky panel with rank for the selected provider across ALL metrics
    in Quarter+Domain (Region ignored). Rendered as a single vertical column."""
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

    rows = []
    for m in metrics_all:
        r = scope[(scope[METRIC] == m) & (scope[PROV_CODE] == provider_code)]
        if r.empty:
            rank_disp = "---"
            pct_disp = "---"
        else:
            rr = r.iloc[0]
            rank_disp = "—" if pd.isna(rr[RANK]) else f"{int(rr[RANK])}"
            pct_disp  = format_percent_display(rr["Percent"], rr[METRIC])

        rows.append(
            '<div class="metric-item">'
            f'  <div class="metric-name">{html.escape(str(m))}</div>'
            '  <div class="metric-row two">'
            f'    <div class="metric-rank">{rank_disp}</div>'
            f'    <div class="metric-pct">{pct_disp}</div>'
            '  </div>'
            '</div>'
        )

    panel_html = (
        '<div class="metrics-panel">'
        f'<div class="metrics-panel-title">{html.escape(panel_title)}</div>'
        + "".join(rows) +
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)   # DO NOT use st.write or st.code


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


# ===================== Sidebar (Upload + Filters) =============
with st.sidebar:
    st.header("📁 Data")

    if "quarterly_bytes" not in st.session_state:
        up = st.file_uploader(
            "Upload the quarterly CSV (columns A–L).",
            type=["csv"],
            key="quarterly_uploader",
            help="Upload once — it will persist while this app is open."
        )
        if up is not None:
            st.session_state["quarterly_bytes"] = up.getvalue()
            st.session_state["quarterly_name"]  = up.name
            st.rerun()
    else:
        st.caption(f"Using: {st.session_state.get('quarterly_name', '(uploaded)')}")
        if st.button("Clear file", key="clear_quarterly"):
            st.session_state.pop("quarterly_bytes", None)
            st.session_state.pop("quarterly_name",  None)
            st.rerun()

# Stop if no file is active
if "quarterly_bytes" not in st.session_state:
    info_banner("👋 Upload a CSV in the sidebar to begin.")
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
quarter = st.sidebar.selectbox("Quarter", quarter_options, index=quarter_options.index(default_quarter) if default_quarter else 0)

df_q = df[df[QUARTER] == quarter]

# 2) Domain (custom order instead of alphabetical)
domain_options = sorted(df_q[DOMAIN].dropna().unique().tolist(), key=lambda d: DOMAIN_ORDER.get(d, 999))
domain = st.sidebar.selectbox("Domain", domain_options)
df_qd = df_q[df_q[DOMAIN] == domain]

# 3) Metric (depends on Domain)
metric_options = sorted(df_qd[METRIC].dropna().unique().tolist())
metric = st.sidebar.selectbox("Metric", metric_options)
df_qdm = df_qd[df_qd[METRIC] == metric]

# 4) Region (optional; no selection = all)
region_options = ["(All Regions)"] + sorted(df_qdm[REGION].dropna().unique().tolist())
region_choice = st.sidebar.selectbox("Region", region_options)
region_selected = None if region_choice == "(All Regions)" else region_choice
df_qdmr = df_qdm if region_selected is None else df_qdm[df_qdm[REGION] == region_selected]

# 5) Provider (optional; filtered by Quarter+Domain+Metric and Region if set)
prov_opts_labels = provider_options(df_qdmr)
default_provider_label = next((lbl for lbl in prov_opts_labels if extract_code_from_label(lbl) == DEFAULT_PROVIDER_CODE), None)
provider_label = st.sidebar.selectbox(
    "Provider (optional)",
    options=["(None)"] + prov_opts_labels,
    index=(0 if default_provider_label is None else (prov_opts_labels.index(default_provider_label) + 1)),
    help="Selecting a provider highlights it and shows KPIs."
)
provider_code = None if provider_label == "(None)" else extract_code_from_label(provider_label)


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
    f'Showing <b>{domain}</b> → <b>{metric}</b> in <b>{quarter}</b>'
    + (f' for <b>{region_selected}</b> region' if region_selected else ' across <b>all regions</b>')
    + '.'
)
st.markdown(f'<div id="context-banner">{context_html}</div>', unsafe_allow_html=True)

# ===================== Provider heading ====================
if provider_code:
    prov_name = provider_name_from_code(df_qdmr, provider_code)
    st.markdown(f"<h2 class='kpi-heading'>{provider_code} — {prov_name}</h2>", unsafe_allow_html=True)
else:
    st.info("Select a provider to see KPIs and trend.")

# ===================== KPI cards (no deltas on Quarterly) ===
if provider_code:
    row = df_qdmr.loc[df_qdmr[PROV_CODE] == provider_code]
    if not row.empty:
        r = row.iloc[0]
        rank_disp = "—" if pd.isna(r[RANK]) else f"{int(r[RANK]):,}"
        num_disp  = "—" if pd.isna(r[NUM])  else f"{int(r[NUM]):,}"
        den_disp  = "—" if pd.isna(r[DEN])  else f"{int(r[DEN]):,}"
        pct_disp  = format_percent_display(r["Percent"], r[METRIC])
        cov_disp  = r[COVERED_MONTHS].strip() if isinstance(r[COVERED_MONTHS], str) and r[COVERED_MONTHS].strip() else "—"

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("Rank", rank_disp)
        with c2: kpi_card("Numerator", num_disp)
        with c3: kpi_card("Denominator", den_disp)
        with c4: kpi_card("% Value", pct_disp)
        with c5: kpi_card("Covered_Months", cov_disp)
    else:
        st.warning("No data for the selected provider under the current Metric/Region.")

# small spacer before the chart + RHS panel
st.markdown('<div class="vgap-3"></div>', unsafe_allow_html=True)

# ===================== Chart (left) + RHS panel (right) =======
left, right = st.columns([0.79, 0.21], gap="large")

with left:
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
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="rhs-sticky">', unsafe_allow_html=True)
    render_metric_rank_panel(df, provider_code, quarter, domain, panel_title=RHS_PANEL_TITLE)
    st.markdown('</div>', unsafe_allow_html=True)


# ===================== Table (full width) =====================
with st.expander("See filtered data as a table and download"):
    # Only the 12 A–L columns (underscored)
    table_cols = [QUARTER, DOMAIN, METRIC, REGION, PROV_CODE, PROV_NAME, NUM, DEN, PCT_STR, RANK, MONTHS_COV, COVERED_MONTHS]
    table_df = df_qdmr[table_cols].copy()
    table_df[PCT_STR] = [format_percent_display(p, m) for p, m in zip(df_qdmr["Percent"], df_qdmr[METRIC])]
    st.dataframe(table_df, hide_index=True, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download CSV", data=make_download_bytes(table_df, False),
                           file_name="filtered_data.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("Download Excel", data=make_download_bytes(table_df, True),
                           file_name="filtered_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

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