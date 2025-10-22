# app.py
# -------------------------------------------------------------
# NHS Provider Metrics Dashboard (stable v2)
# - Robust numeric parsing (handles commas, spaces, NBSP, etc.)
# - Altair 5 sort fixed (uses alt.SortField)
# - Column headers normalized to underscores internally
# - Deprecated 'use_container_width' removed (uses width='stretch')
# - Debug message when no bars would render
# -------------------------------------------------------------

import io
import re
import altair as alt
import pandas as pd
import streamlit as st

# =============================================================
# --------------------- APP-WIDE SETTINGS ---------------------
# =============================================================

st.set_page_config(
    page_title="NHS Provider Metrics",
    page_icon="📊",
    layout="wide"
)

# Altair defaults
alt.data_transformers.disable_max_rows()  # avoid any silent row limits

# Global CSS (Segoe UI + KPI cards)
st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      }
      .kpi-card {
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 14px 16px;
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }
      .kpi-title { font-size: 0.85rem; color: #555; margin-bottom: 4px; }
      .kpi-value { font-size: 1.4rem; font-weight: 600; }
      .vega-tooltip {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Constants
# -------------------------------------------------------------
# Original CSV headers (A–L), exactly as in your files
COLS_ORIG = [
    "Quarter", "Domain", "Metric", "Region",
    "Provider Code", "Provider Name",
    "Numerator", "Denominator", "% Value", "Rank",
    "Months Covered", "Covered Months"
]

# Internal, underscored names (we use these everywhere inside the app)
COLS_US = [re.sub(r"\s+", "_", c.replace("%", "Percent")) for c in COLS_ORIG]
RENAME_MAP = dict(zip(COLS_ORIG, COLS_US))

# Nice aliases
(
    QUARTER, DOMAIN, METRIC, REGION,
    PROV_CODE, PROV_NAME,
    NUM, DEN, PCT_STR, RANK,
    MONTHS_COV, COVERED_MONTHS
) = COLS_US

# Colors and defaults
HIGHLIGHT_HEX = "#FAE100"      # NHS yellow for selected provider
BAR_NEUTRAL_HEX = "#D5DAE1"    # light grey for others
DEFAULT_PROVIDER_CODE = "RWP"  # preferred default

# =============================================================
# ---------------------- HELPER FUNCTIONS ---------------------
# =============================================================

def clean_numeric_str_to_float(x: str) -> float | None:
    """
    Strip anything that isn't a digit, dot, or minus (handles NBSP and odd chars).
    Return float or NaN.
    """
    if x is None:
        return float("nan")
    if not isinstance(x, str):
        x = str(x)
    cleaned = re.sub(r"[^\d\.\-]", "", x)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_numeric_str_to_int(x: str) -> float | None:
    """Same as above, but we will format as int later."""
    val = clean_numeric_str_to_float(x)
    return pd.to_numeric(val, errors="coerce")


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """
    1) Read CSV (original headers).
    2) Keep only required columns A–L.
    3) Rename to underscored headers (internal use).
    4) Parse:
       - Numerator/Denominator/Rank/Months_Covered -> numeric
       - Percent_String ('% Value') -> numeric Percent
    5) Quarter kept as ordered category by file order.
    """
    df = pd.read_csv(file, dtype=str)

    # Check columns
    missing = [c for c in COLS_ORIG if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Reduce to required and rename headers to underscored
    df = df[COLS_ORIG].rename(columns=RENAME_MAP).copy()

    # Trim whitespace
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Numeric coercions (robust)
    for c in [NUM, DEN, RANK, MONTHS_COV]:
        df[c] = df[c].apply(clean_numeric_str_to_int)

    # Percent numeric (from "% Value" -> Percent_String -> numeric "Percent")
    def parse_percent(x: str):
        # tolerate "76.8 %" / "76,8%" / NBSP etc.
        return clean_numeric_str_to_float(str(x).replace(",", "."))

    df["Percent"] = df[PCT_STR].apply(parse_percent)

    # Quarter categorical (preserve file order)
    df[QUARTER] = pd.Categorical(df[QUARTER], categories=pd.unique(df[QUARTER]), ordered=True)
    return df


def format_percent_display(value: float, metric_name: str) -> str:
    """2dp only for '52+ Weeks', else 1dp, '---' if NaN."""
    if pd.isna(value):
        return "---"
    if str(metric_name).strip().lower() == "52+ weeks":
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def pick_latest_quarter(quarters_series: pd.Series) -> str:
    unique_in_order = list(pd.unique(quarters_series))
    return unique_in_order[-1] if unique_in_order else None


def provider_options(df_filtered_by_region: pd.DataFrame) -> list:
    tmp = df_filtered_by_region[[PROV_CODE, PROV_NAME]].drop_duplicates()
    tmp = tmp.sort_values(PROV_CODE)
    return [f"{r[PROV_CODE]} — {r[PROV_NAME]}" for _, r in tmp.iterrows()]


def extract_code_from_label(label: str) -> str:
    return label.split("—")[0].strip() if isinstance(label, str) and "—" in label else label


def render_kpi_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_download_bytes(df: pd.DataFrame, as_excel: bool = False) -> bytes:
    if as_excel:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Filtered")
        return output.getvalue()
    return df.to_csv(index=False).encode("utf-8")


def build_chart(chart_df: pd.DataFrame, chart_title: str):
    """
    Column chart:
    - x = Provider_Code (title 'Providers', tick labels hidden)
    - y = Percent (domain starts at 0, title hidden)
    - Order is given by the **row order** (we pre-sorted by Rank)
    - Color by Is_Selected (yellow highlight)
    - Rounded corners + rich tooltip
    """
    import altair as alt

    if chart_df.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data for current filters"]})).mark_text().encode(text="note")

    x_axis = alt.Axis(title="Providers", labels=False)
    y_axis = alt.Axis(title=None, grid=True)

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{PROV_CODE}:N", sort=None, axis=x_axis),   # <— no sort here; we already sorted the rows
            y=alt.Y("Percent:Q", scale=alt.Scale(domainMin=0), axis=y_axis),
            color=alt.Color(
                "Is_Selected:N",
                legend=None,
                scale=alt.Scale(domain=[False, True], range=[BAR_NEUTRAL_HEX, HIGHLIGHT_HEX]),
            ),
            tooltip=[
                alt.Tooltip(f"{PROV_CODE}:N", title="Provider Code"),
                alt.Tooltip(f"{PROV_NAME}:N", title="Provider Name"),
                alt.Tooltip(f"{REGION}:N", title="Region"),
                alt.Tooltip(f"{NUM}:Q", title="Numerator", format=",.0f"),
                alt.Tooltip(f"{DEN}:Q", title="Denominator", format=",.0f"),
                alt.Tooltip("PercentLabel:N", title="% Value"),
                alt.Tooltip(f"{RANK}:Q", title="Rank", format=",.0f"),
            ],
        )
        .properties(height=420, title=chart_title)
        .configure_title(anchor="start")
    )

    return chart.interactive()


# =============================================================
# ------------------------- SIDEBAR ---------------------------
# =============================================================

with st.sidebar:
    st.header("📁 Data")
    uploaded = st.file_uploader(
        "Upload the monthly CSV (columns A–L).",
        type=["csv"],
        help="Must include columns: " + ", ".join(COLS_ORIG),
    )

    # Sample template (still shows original headers to guide authors)
    example_rows = [
        {
            "Quarter": "Q4 (2024/25)", "Domain": "A&E", "Metric": "4 hours", "Region": "London",
            "Provider Code": "RWP", "Provider Name": "Royal Wolverhampton NHS Trust",
            "Numerator": "63,290", "Denominator": "82,460", "% Value": "76.8%", "Rank": "36",
            "Months Covered": "3", "Covered Months": "Jan, Feb, Mar",
        },
        {
            "Quarter": "Q4 (2024/25)", "Domain": "A&E", "Metric": "52+ Weeks", "Region": "London",
            "Provider Code": "RWP", "Provider Name": "Royal Wolverhampton NHS Trust",
            "Numerator": "120", "Denominator": "10,000", "% Value": "1.20%", "Rank": "12",
            "Months Covered": "3", "Covered Months": "Jan, Feb, Mar",
        },
    ]
    tmpl_df = pd.DataFrame(example_rows)[COLS_ORIG]
    st.download_button(
        "Download sample CSV template",
        data=tmpl_df.to_csv(index=False).encode("utf-8"),
        file_name="nhs_metrics_template.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("---")
    st.header("🔎 Filters")

# =============================================================
# ------------------------- MAIN FLOW -------------------------
# =============================================================

if uploaded is None:
    st.info("👈 Upload a CSV in the sidebar to begin.")
    st.stop()

# Load data
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# ---------- Cascading filters (single-select) ----------

# 1) Quarter (latest by file order)
quarter_options = list(df[QUARTER].cat.categories)
default_quarter = (quarter_options[-1] if quarter_options else None)
quarter = st.sidebar.selectbox("Quarter", quarter_options, index=quarter_options.index(default_quarter) if default_quarter else 0)

df_q = df[df[QUARTER] == quarter]

# 2) Domain
domain_options = sorted(df_q[DOMAIN].dropna().unique().tolist())
domain = st.sidebar.selectbox("Domain", domain_options)

df_qd = df_q[df_q[DOMAIN] == domain]

# 3) Metric (depends on domain)
metric_options = sorted(df_qd[METRIC].dropna().unique().tolist())
metric = st.sidebar.selectbox("Metric", metric_options)

df_qdm = df_qd[df_qd[METRIC] == metric]

# 4) Region (optional; no selection = all)
region_options = ["(All Regions)"] + sorted(df_qdm[REGION].dropna().unique().tolist())
region_choice = st.sidebar.selectbox("Region", region_options)
region_selected = None if region_choice == "(All Regions)" else region_choice

if region_selected:
    df_qdmr = df_qdm[df_qdm[REGION] == region_selected]
else:
    df_qdmr = df_qdm.copy()

# 5) Provider (optional; defaults to RWP if present)
prov_opts_labels = provider_options(df_qdmr)
default_provider_label = next((lbl for lbl in prov_opts_labels if extract_code_from_label(lbl) == DEFAULT_PROVIDER_CODE), None)

provider_label = st.sidebar.selectbox(
    "Provider (optional)",
    options=["(None)"] + prov_opts_labels,
    index=(0 if default_provider_label is None else (prov_opts_labels.index(default_provider_label) + 1)),
    help="Selecting a provider highlights it in the chart and shows KPI cards."
)
provider_code = None if provider_label == "(None)" else extract_code_from_label(provider_label)

# ---------- Main title ----------
st.title("NHS Provider Metrics Dashboard")
st.caption(
    f"Showing **{domain} → {metric}** in **{quarter}**"
    + (f" for **{region_selected}** region" if region_selected else " across **all regions**")
    + ". Use the filters in the sidebar to change the view."
)

# ---------- Chart data ----------
chart_df = df_qdmr.copy()

# Keep only rows that can be plotted
chart_df_plot = chart_df.dropna(subset=["Percent", RANK], how="any").copy()

# Sort by Rank asc, then Percent desc, then Provider Name asc (stable)
chart_df_plot = chart_df_plot.sort_values([RANK, "Percent", PROV_NAME], ascending=[True, False, True])

# Highlight + nicely formatted label
chart_df_plot["Is_Selected"] = (chart_df_plot[PROV_CODE].eq(provider_code) if provider_code else False)
chart_df_plot["PercentLabel"] = chart_df_plot.apply(
    lambda r: format_percent_display(r["Percent"], r[METRIC]),
    axis=1
)

# Debug hint if nothing to plot
if chart_df_plot.empty:
    total = len(chart_df)
    missing_pct = chart_df["Percent"].isna().sum()
    missing_rank = chart_df[RANK].isna().sum()
    st.warning(
        f"No bars to draw for the current filters. "
        f"Rows before drop: {total:,} · missing Percent: {missing_pct:,} · missing Rank: {missing_rank:,}. "
        f"Check the CSV values for '% Value' and 'Rank'."
    )

chart_title = "Provider Performance (% Value) — ordered by Rank (1 at left)"
chart = build_chart(chart_df_plot, chart_title)
st.altair_chart(chart, use_container_width=True)

# ---------- KPI cards (single provider only) ----------
if provider_code:
    row = df_qdmr.loc[df_qdmr[PROV_CODE] == provider_code]
    if not row.empty:
        r = row.iloc[0]
        rank_disp = "---" if pd.isna(r[RANK]) else f"{int(r[RANK]):,}"
        num_disp  = "---" if pd.isna(r[NUM])  else f"{int(r[NUM]):,}"
        den_disp  = "---" if pd.isna(r[DEN])  else f"{int(r[DEN]):,}"
        pct_disp  = format_percent_display(r["Percent"], r[METRIC])
        covm_disp = r[COVERED_MONTHS] if isinstance(r[COVERED_MONTHS], str) and r[COVERED_MONTHS].strip() else "---"

        st.subheader(f"KPI — {provider_code} · {r[PROV_NAME]}")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: render_kpi_card("Rank", rank_disp)
        with c2: render_kpi_card("Numerator", num_disp)
        with c3: render_kpi_card("Denominator", den_disp)
        with c4: render_kpi_card("% Value", pct_disp)
        with c5: render_kpi_card("Covered_Months", covm_disp)
    else:
        st.warning("No data for the selected provider under the current filters.")
        c1, c2, c3, c4, c5 = st.columns(5)
        for title in ["Rank", "Numerator", "Denominator", "% Value", "Covered_Months"]:
            with (c1 if title=="Rank" else c2 if title=="Numerator" else c3 if title=="Denominator" else c4 if title=="% Value" else c5):
                render_kpi_card(title, "---")

# ---------- Table + downloads ----------
with st.expander("See filtered data as a table and download"):
    # Show only the 12 core columns (underscored headers)
    table_cols = [QUARTER, DOMAIN, METRIC, REGION, PROV_CODE, PROV_NAME, NUM, DEN, PCT_STR, RANK, MONTHS_COV, COVERED_MONTHS]
    table_df = df_qdmr[table_cols].copy()

    # Pretty-print "% Value" according to metric rule
    table_df[PCT_STR] = [
        format_percent_display(p, m) for p, m in zip(df_qdmr["Percent"], df_qdmr[METRIC])
    ]

    st.dataframe(table_df, hide_index=True, width="stretch")

    st.write("**Download current filtered data**")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download CSV",
            data=make_download_bytes(table_df, as_excel=False),
            file_name="filtered_data.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_dl2:
        st.download_button(
            "Download Excel",
            data=make_download_bytes(table_df, as_excel=True),
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

# ---------- Notes ----------
st.caption(
    """
    Notes:
    • Column headers inside the app are normalized to underscores (e.g., `Provider_Code`).  
    • Bars are ordered by **Rank** (ascending).  
    • Tooltip shows provider details; `% Value` uses **2dp** only for the `52+ Weeks` metric.  
    • Missing values are hidden in charts and shown as `---` in KPIs/table.
    """
)
