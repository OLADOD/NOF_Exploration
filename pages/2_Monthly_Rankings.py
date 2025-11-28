# pages/2_Monthly_Rankings.py
# ------------------------------------------------------------------
# NOF Monthly Rankings Dashboard
# Data columns expected:
#   Month, Domain, Metric, Region, Provider_Code, Provider_Name,
#   Numerator, Denominator, %_Value, Rank, Rank_Region,
#   Region_Size, Data_Date_Used, Notes_Flags
# ------------------------------------------------------------------

import io
import re
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd, io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import streamlit as st
import html
from pathlib import Path

# add (or keep) this import in the file header
from decimal import Decimal, ROUND_HALF_UP

# -------------------------- Page setup ----------------------------
st.set_page_config(
    page_title="NOF Monthly Rankings",   # or Quarterly
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"       # <— add this
)

# Home owns this flag. Pages must not set a default.
st.session_state.setdefault("remember_filters", True)
REMEMBER = st.session_state.get("remember_filters", True)

# ---- Global page CSS (inline; keeps sidebar; pulls content up) ----

def use_ui_css():
    p = Path(__file__).parents[1] / "assets" / "ui.css"
    if p.exists():
        css = p.read_text(encoding="utf-8")
        # add a tiny cache-buster comment using mtime so browser applies updates
        css += f"\n/* mtime:{int(p.stat().st_mtime)} */"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

use_ui_css()


# --- Title + Logo helpers (monthly) -----------------------------------------

ROOT_ASSETS  = Path(__file__).parents[1] / "assets"
PAGES_ASSETS = Path(__file__).parent / "assets"
for candidate in (ROOT_ASSETS / "NOF_Logo.svg", PAGES_ASSETS / "NOF_Logo.svg"):
    if candidate.is_file():
        LOGO_FILE = candidate; break
else:
    LOGO_FILE = None

def read_svg_file(p: Path) -> str:
    try: return p.read_text(encoding="utf-8")
    except Exception: return ""
logo_svg = read_svg_file(LOGO_FILE) if LOGO_FILE else ""


# -------------------------- Resuse Filters  ---------------------------

# Reuse the “remember filters across pages” if set on home
REMEMBER = st.session_state.get("remember_filters", True)

# Shared palette (matches the quarterly look)
NHS_YELLOW = "#FAE100"
BAR_GREY   = "#D5DAE1"
CARD_BG = "#F5FBFB"


# ---------------------------- Styles ------------------------------
st.markdown("""
<style>


/* Small helper for muted text */
.muted { color:#6B7280; }

</style>
""", unsafe_allow_html=True)


# ----------------------- Expected columns -------------------------
COLS = [
    "Month", "Domain", "Metric", "Region",
    "Provider_Code", "Provider_Name",
    "Numerator", "Denominator", "%_Value", "Rank",
    "Rank_Region", "Region_Size", "Data_Date_Used", "Notes_Flags"
]

# --------------------- Data cleaning helpers ----------------------
def to_int_or_float(x: str):
    """Convert to numeric, preserving decimals if present."""
    if pd.isna(x): 
        return float("nan")
    if not isinstance(x, str):
        x = str(x)
    # Replace comma decimal separator with period, remove other non-numeric chars except decimal point and minus
    x = x.replace(",", ".")  # Handle European format
    cleaned = re.sub(r"[^\d\.\-]", "", x)  # Keep digits, decimal point, and minus sign
    return pd.to_numeric(cleaned, errors="coerce")

def to_float(x: str) -> float:
    """Convert strings like '76.8%' or '76,8' to a float (percent points)."""
    if pd.isna(x): return float("nan")
    if not isinstance(x, str): x = str(x)
    # Replace comma decimal and strip % sign/commas
    x = x.replace(",", ".")
    x = re.sub(r"[^\d\.\-]", "", x)
    return pd.to_numeric(x, errors="coerce")

def format_percent_display(value, metric_name: str) -> str:
    """Match Quarterly: 2dp only for '52+ weeks', else 1dp; '—' if NaN."""
    import pandas as pd
    if pd.isna(value):
        return "—"
    return f"{value:.2f}%" if str(metric_name).strip().lower() == "52+ weeks" else f"{value:.1f}%"

def format_whole_round(x) -> str:
    """Quarterly-style integer with true half-up rounding + thousands separators."""
    import pandas as pd
    if pd.isna(x): return "—"
    try:
        q = Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return f"{int(q):,}"
    except Exception:
        return "—"

def format_region_rank(rank_val, size_val) -> str:
    """e.g., '3 out of 19' or '—' if missing."""
    import pandas as pd
    if pd.isna(rank_val) or pd.isna(size_val):
        return "—"
    return f"{int(rank_val)} out of {int(size_val)}"

def format_overall_rank_text(rank_val, nat_total) -> str:
    """Plain text 'X out of Y' or '—' if missing."""
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)} out of {int(nat_total)}"

def format_overall_rank_html(rank_val, nat_total) -> str:
    import pandas as pd
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)}<span class='muted'> out of {int(nat_total)}</span>"


@st.cache_data(show_spinner=False)
def load_monthly_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Read monthly CSV from uploaded bytes; return a clean DataFrame.
    Cached by the byte content.
    """
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Keep only expected columns and strip whitespace
    df = df[COLS].copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Robust Month parse: try US (mm/dd) first, then UK (dd/mm); normalise to month start
    dt_us = pd.to_datetime(df["Month"], dayfirst=False, errors="coerce")
    dt_uk = pd.to_datetime(df["Month"], dayfirst=True,  errors="coerce")

    # pick the parse with more valid dates
    dt = dt_us if dt_us.notna().sum() >= dt_uk.notna().sum() else dt_uk
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Month values could not be parsed in {bad} rows.")

    # normalise to the first day of the month and create a friendly label
    df["Month_dt"]   = dt.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1) + pd.offsets.Day(1)
    df["Month_disp"] = df["Month_dt"].dt.strftime("%b-%y")  # e.g. Jan-25

    # Numeric columns
    # Numeric columns - use to_int_or_float to preserve decimals
    for col in ["Numerator", "Denominator", "Rank", "Rank_Region", "Region_Size"]:
        df[col] = df[col].apply(to_int_or_float)

    # FIXED: Percent column handling - convert per-row based on value range
    df["Percent"] = df["%_Value"].apply(to_float)
    
    # Check each value individually: if it looks like a ratio (0-1), convert to percentage points
    def convert_to_percentage(val):
        if pd.isna(val):
            return val
        # If value is between 0 and 1 (exclusive of exactly 0 and 1), treat as ratio
        if 0 < val <= 1.0:
            return val * 100.0
        # If value is already in percentage range or is 0, keep as-is
        return val
    
    df["Percent"] = df["Percent"].apply(convert_to_percentage)

    return df

def weighted_percentage(sub: pd.DataFrame,
                        num_col: str = "Numerator",
                        den_col: str = "Denominator") -> float:
    """
    Weighted % computed ONLY from counts:
        100 * (Σ numerator) / (Σ denominator)

    - Coerces to numeric in case the caller passes an unclean slice.
    - Ignores NaNs.
    - Returns NaN if denominator sum is 0.
    - Clips to [0, 100] to avoid 100.0001% due to float rounding.
    """
    s_num = pd.to_numeric(sub[num_col], errors="coerce")
    s_den = pd.to_numeric(sub[den_col], errors="coerce")

    num = s_num.fillna(0).sum()
    den = s_den.fillna(0).sum()

    if not den or np.isnan(den):
        return float("nan")

    pct = 100.0 * num / den
    # guard against tiny floating error
    return float(np.clip(pct, 0.0, 100.0))

def delta_display(curr: Optional[float], prev: Optional[float], higher_is_better: bool) -> Tuple[str, str]:
    """
    Compute a signed change and classify as good/bad/neutral for coloring.
    Returns: (text, css_class)
    - For Ranks: LOWER is better (higher_is_better=False).
    - For % Value: HIGHER is better (higher_is_better=True).
    - For Numerator/Denominator: neutral coloring (we'll set higher_is_better=None).
    """
    if curr is None or pd.isna(curr) or prev is None or pd.isna(prev):
        return "—", "neu"

    change = curr - prev
    if abs(change) < 1e-9:
        return "0", "neu"

    if higher_is_better is None:
        # numeric counts—just show sign without good/bad semantics
        return f"{change:+,.0f}", "neu"

    # Good if it moves in the desired direction
    is_good = (change > 0) if higher_is_better else (change < 0)
    css = "good" if is_good else "bad"

    # For percentages: 1 decimal; for ranks: no decimals
    if higher_is_better:
        # percentage points
        return f"{change:+.1f} pp", css
    else:
        # ranks
        return f"{change:+.0f}", css

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


def info_banner(msg: str):
    st.markdown(
        f"<div class='info-row'><div class='info-banner'>{msg}</div></div>",
        unsafe_allow_html=True,
    )

def kpi_card(title: str, value_html: str, delta_html: str | None = None, delta_class: str = "neu"):
    """Render a KPI card. If delta_html is None, no delta line is shown."""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value_html}</div>
          {f"<div class='kpi-delta {delta_class}'>{delta_html}</div>" if delta_html is not None else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

def provider_name_from_code(df_scope, code: str) -> str:
    if not code: return ""
    s = df_scope.loc[df_scope["Provider_Code"] == code, "Provider_Name"].dropna()
    return s.iloc[0] if len(s) else ""

def sanity_check(scope: pd.DataFrame) -> None:
    # counts-based
    pct_counts = weighted_percentage(scope)
    # if Percent column is 0–100, convert to proportion
    p = pd.to_numeric(scope["Percent"], errors="coerce")
    prop = p / 100.0
    w_avg = np.average(prop.dropna(), weights=pd.to_numeric(scope["Denominator"], errors="coerce").reindex(prop.index).fillna(0))
    pct_from_percent = 100.0 * w_avg
    st.caption(f"Debug · counts={pct_counts:.2f}%, weighted(Percent)={pct_from_percent:.2f}%")

# sanity_check(comp_scope)  # uncomment when debugging

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


# ------------------------ End of def -----------------------------

# ------------------------- Sidebar upload -------------------------
with st.sidebar:
    st.header("📁 Monthly data")

    has_monthly = ("monthly_bytes" in st.session_state)

    if has_monthly:
        st.caption(f"Using: {st.session_state.get('monthly_name', '(uploaded)')}")
        if st.button("Clear file", key="clear_monthly"):
            st.session_state.pop("monthly_bytes", None)
            st.session_state.pop("monthly_name",  None)
            st.rerun()
    else:
        st.warning("No Monthly CSV loaded. Go to the Homepage to upload.", icon="⚠️")
        if st.button("Go to Homepage", key="goto_home_for_monthly"):
            st.switch_page("Homepage.py")
    # ← No st.stop() here


# If no file yet → show empty state and stop (Monthly)
if "monthly_bytes" not in st.session_state:
    render_empty_state(
        page_name="Monthly",
        template_cols=[
            "Month","Domain","Metric","Region",
            "Provider Code","Provider Name","Numerator","Denominator",
            "% Value","Rank","Rank_Region","Region_Size","Data_Date_Used",
        ],
        demo_rel_path=None
    )
    st.stop()

try:
    df = load_monthly_csv(st.session_state["monthly_bytes"])
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# --------------------------- Filters ------------------------------
# 1) Month (latest by date)
months_sorted = df.sort_values("Month_dt")["Month_disp"].unique().tolist()
default_month = months_sorted[-1]

# Read remembered selection if enabled
def remembered(key: str, default):
    if REMEMBER and key in st.session_state:
        return st.session_state[key]
    return default

month = st.sidebar.selectbox("Month", months_sorted, index=months_sorted.index(remembered("m_month", default_month)))

# Build a lookup from display label -> datetime
_month_map = (df[["Month_disp","Month_dt"]]
              .drop_duplicates("Month_disp")
              .set_index("Month_disp")["Month_dt"]
              .to_dict())

# Long month label for context banner (e.g., January-2025)
month_long = _month_map.get(month, pd.NaT)
month_long = month_long.strftime("%B-%Y") if pd.notna(month_long) else month  # fallback to original if not found


df_m = df[df["Month_disp"] == month]

# 2) Domain (custom order)
DOMAIN_ORDER = {"A&E": 0, "Cancer": 1, "RTT": 2, "Diagnostic": 3}
domain_opts = sorted(df_m["Domain"].dropna().unique().tolist(), key=lambda d: DOMAIN_ORDER.get(d, 999))
domain = st.sidebar.selectbox("Domain", domain_opts, index=domain_opts.index(remembered("m_domain", domain_opts[0])))

df_md = df_m[df_m["Domain"] == domain]

# 3) Metric (depends on Domain)
metric_opts = sorted(df_md["Metric"].dropna().unique().tolist())
metric_rem  = remembered("m_metric", metric_opts[0])
if metric_rem not in metric_opts:
    metric_rem = metric_opts[0]
metric = st.sidebar.selectbox("Metric", metric_opts, index=metric_opts.index(metric_rem))


df_mdm = df_md[df_md["Metric"] == metric]

# 4) Region (optional; no selection => all)
region_opts = ["(All Regions)"] + sorted(df_mdm["Region"].dropna().unique().tolist())
region_choice = remembered("m_region", "(All Regions)")
if region_choice not in region_opts:
    region_choice = "(All Regions)"
region = st.sidebar.selectbox("Region", region_opts, index=region_opts.index(region_choice))
region_selected = None if region == "(All Regions)" else region
df_mdmr = df_mdm if region_selected is None else df_mdm[df_mdm["Region"] == region_selected]

# 5) Provider (optional; filtered by Month+Domain+Metric+(Region))
def provider_labels(sub: pd.DataFrame) -> list:
    tmp = sub[["Provider_Code","Provider_Name"]].drop_duplicates().sort_values("Provider_Code")
    return [f"{r.Provider_Code} — {r.Provider_Name}" for _, r in tmp.iterrows()]

prov_labels = provider_labels(df_mdmr)
DEFAULT_PROVIDER_CODE = "RWP"
default_label = next((lbl for lbl in prov_labels
                      if lbl.split("—")[0].strip() == DEFAULT_PROVIDER_CODE), None)

# NEW: read remembered local label safely (fixes undefined prov_label_rem)
prov_label_rem = st.session_state.get("m_provider", "(None)")

# Check if there's a shared provider selection from another page
shared_provider = st.session_state.get("shared_provider_code", None) if REMEMBER else None

# Compute the default index robustly:
# - If a shared provider exists AND is available under current filters, use it.
# - Else if a remembered local provider label is valid, use that.
# - Else if the hard-coded default exists, use that.
# - Else fall back to "(None)".
# Work out the default index clearly and safely
if shared_provider and any(lbl.split("—")[0].strip() == shared_provider for lbl in prov_labels):
    # Find the label that matches the shared provider code
    matching_label = next(lbl for lbl in prov_labels if lbl.split("—")[0].strip() == shared_provider)
    # IMPORTANT: we're searching in ["(None)"] + prov_labels, so do NOT add +1 again
    index_default = (["(None)"] + prov_labels).index(matching_label)
elif prov_label_rem in (["(None)"] + prov_labels):
    index_default = (["(None)"] + prov_labels).index(prov_label_rem)
elif default_label:
    index_default = 1 + prov_labels.index(default_label)  # +1 for "(None)" at position 0
else:
    index_default = 0

provider_label = st.sidebar.selectbox("Provider (optional)",
                                      options=["(None)"] + prov_labels,
                                      index=index_default)
provider_code = None if provider_label == "(None)" else provider_label.split("—")[0].strip()

# Save local page memory (unchanged)
if REMEMBER:
    st.session_state.update({
        "m_month": month, "m_domain": domain, "m_metric": metric,
        "m_region": region, "m_provider": provider_label
    })

# NEW: Only update the shared provider when a real provider is selected AND remember is ON.
if REMEMBER and provider_code:
    st.session_state["shared_provider_code"] = provider_code
# If "(None)" is selected, do NOT clear the shared code; leave it as-is.

# --------------------------- Header -------------------------------
st.markdown(
    f"""
    <div class="app-header" role="banner" aria-label="Header">
      <div class="app-logo" aria-hidden="true">{logo_svg}</div>
      <h1 id="page-title">NOF Monthly Rankings</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== Context banner ======================
context_html = (
    f'Showing <b>{html.escape(domain)}</b> → <b>{html.escape(metric)}</b> in <b>{html.escape(month_long)}</b>'
    + (f' for <b>{html.escape(region_selected)}</b> region' if region_selected else ' across <b>all regions</b>')
    + '.'
)
st.markdown(f'<div id="context-banner">{context_html}</div>', unsafe_allow_html=True)

# ===================== Provider heading ====================
if provider_code:
    prov_name = provider_name_from_code(df_mdmr, provider_code)  # Month+Domain+Metric(+Region)
    st.markdown(f"<h2 class='kpi-heading'>{html.escape(prov_name)} ({html.escape(provider_code)})</h2>", unsafe_allow_html=True)
else:
    st.info("Select a provider to see KPIs and trend.")

# ===================== KPI cards (with deltas on Monthly) ===
if provider_code:
    row = df_mdmr.loc[df_mdmr["Provider_Code"] == provider_code]
    if not row.empty:
        r = row.iloc[0]

        # Current values
        rank_val     = r["Rank"]
        rank_reg_val = r["Rank_Region"]
        region_size  = r["Region_Size"]
        num_val      = r["Numerator"]
        den_val      = r["Denominator"]
        pct_val      = r["Percent"]
        as_of        = r["Data_Date_Used"]

        # Previous month for same Provider+Domain+Metric
        prev_month_dt = (r["Month_dt"] - pd.offsets.MonthBegin(1))
        prev = df[(df["Provider_Code"] == provider_code)
                  & (df["Domain"] == domain)
                  & (df["Metric"] == metric)
                  & (df["Month_dt"] == prev_month_dt)]
        if not prev.empty:
            p = prev.iloc[0]
            rank_prev, rank_reg_prev = p["Rank"], p["Rank_Region"]
            num_prev,  den_prev      = p["Numerator"], p["Denominator"]
            pct_prev                 = p["Percent"]
        else:
            rank_prev = rank_reg_prev = num_prev = den_prev = pct_prev = None

        # Deltas (Rank lower-is-better; % higher-is-better)
        d_rank,  cls_rank  = delta_display(rank_val,     rank_prev,     higher_is_better=False)
        d_rankr, cls_rankr = delta_display(rank_reg_val, rank_reg_prev, higher_is_better=False)
        d_num,   cls_num   = delta_display(num_val,      num_prev,      higher_is_better=None)
        d_den,   cls_den   = delta_display(den_val,      den_prev,      higher_is_better=None)
        d_pct,   cls_pct   = delta_display(pct_val,      pct_prev,      higher_is_better=True)

        # -------- Overall Rank as "X out of Y" --------
        nat_n = df_mdm["Provider_Code"].nunique()  # Month+Domain+Metric national total
        overall_html = format_overall_rank_html(r["Rank"], nat_n)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Overall Rank", overall_html, d_rank, cls_rank)

        with c2:
            label = ("—" if pd.isna(rank_reg_val) else f"{int(rank_reg_val)}")
            tail  = ("" if pd.isna(region_size) else f"<span class='muted'> out of {int(region_size)}</span>")
            kpi_card("Region Rank", f"{label}{tail}", d_rankr, cls_rankr)

        with c3:
            kpi_card("Numerator", ("—" if pd.isna(num_val) else f"{int(num_val):,}"), d_num, cls_num)
        with c4:
            kpi_card("Denominator", ("—" if pd.isna(den_val) else f"{int(den_val):,}"), d_den, cls_den)
        with c5:
            val = "—" if pd.isna(pct_val) else (f"{pct_val:.2f}%" if str(metric).strip().lower()=="52+ weeks" else f"{pct_val:.1f}%")
            kpi_card("% Value", val, d_pct, cls_pct)

    else:
        st.warning("No row for that provider under the current month/domain/metric/region.")

st.markdown('<div class="vgap-3"></div>', unsafe_allow_html=True)

# 1) Bar chart by Overall Rank (Region narrows; Provider does NOT filter)
bars_df = df_mdm if region_selected is None else df_mdm[df_mdm["Region"] == region_selected]
bars_df = bars_df.dropna(subset=["Percent","Rank"])
bars_df = bars_df.sort_values(["Rank","Percent","Provider_Name"], ascending=[True, False, True]).copy()
bars_df["Is_Selected"] = bars_df["Provider_Code"].eq(provider_code) if provider_code else False

# Pre-format labels to match Quarterly’s hover
bars_df["PercentLabel"]     = bars_df.apply(lambda r: format_percent_display(r["Percent"], r["Metric"]), axis=1)
bars_df["NumLabel"]         = bars_df["Numerator"].map(format_whole_round)
bars_df["DenLabel"]         = bars_df["Denominator"].map(format_whole_round)
bars_df["RegionRankLabel"]  = bars_df.apply(lambda r: format_region_rank(r["Rank_Region"], r["Region_Size"]), axis=1)

bars_df = bars_df.drop_duplicates(subset=["Provider_Code"])

fig = px.bar(bars_df, x="Provider_Code", y="Percent", title="Provider Performance (% Value) — ordered by Rank (1 at left)")
fig.update_traces(
    marker_color=bars_df["Is_Selected"].map({True: NHS_YELLOW, False: BAR_GREY}),
    customdata=bars_df[[
        "Provider_Code", "Provider_Name", "Region",
        "NumLabel", "DenLabel", "PercentLabel",
        "Rank", "RegionRankLabel"
    ]].values,
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
    template="simple_white", height=410, margin=dict(l=10,r=10,t=50,b=10),
    xaxis_title="Providers", xaxis_showticklabels=False,
    yaxis_title=None, yaxis_range=[0,None], yaxis_ticksuffix="%",
    bargap=0.15, title=dict(x=0)
)

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

st.plotly_chart(fig, use_container_width=True)

# Add a thin separator line above the chart
st.markdown('<hr class="chart-separator">', unsafe_allow_html=True)

# ===================== Trend (left) + Metrics panel (right) =====================
# st.markdown("<hr class='chart-separator cmp-sep'>", unsafe_allow_html=True)

left, right = st.columns([0.60, 0.40], gap="medium")   # trend wider, cards narrower

with left:
    # 2) 24-month trend that IGNORES the Month slicer
    # Always use the full Domain+Metric slice (not month-filtered)
    domain_metric = df[(df["Domain"] == domain) & (df["Metric"] == metric)].copy()

    # If Month_dt isn't datetime yet, make it so (safe no-op if already dt)
    if not pd.api.types.is_datetime64_any_dtype(domain_metric["Month_dt"]):
        domain_metric["Month_dt"] = pd.to_datetime(domain_metric["Month_dt"])

    # Decide which region line to show (avoid month-filtered frames here):
    # - If user picked a Region, use it
    # - Else, if a provider is selected, use that provider's *most recent* Region
    # - Else, no region line
    if region_selected and region_selected != "(All Regions)":
        region_for_compare = region_selected
    elif provider_code:
        prov_rows = domain_metric.loc[domain_metric["Provider_Code"] == provider_code]
        if not prov_rows.empty:
            # use the region from the provider's most recent record in this Domain+Metric
            region_for_compare = prov_rows.sort_values("Month_dt")["Region"].iloc[-1]
        else:
            region_for_compare = None
    else:
        region_for_compare = None

    # Always pick the last 24 unique months up to the *latest* month in the data
    latest_dt = domain_metric["Month_dt"].max()
    uniq_months = (
        domain_metric[["Month_dt"]]
        .drop_duplicates()
        .sort_values("Month_dt")["Month_dt"]
    )

    months_sorted = uniq_months[uniq_months <= latest_dt].tail(24).tolist()

    # Build series for Provider, Region weighted, National weighted
    trend_rows = []
    for mdt in months_sorted:
        scope = domain_metric[domain_metric["Month_dt"] == mdt]

        # Provider % for this month (if selected)
        if provider_code:
            p = scope.loc[scope["Provider_Code"] == provider_code, "Percent"]
            prov = float(p.iloc[0]) if len(p) else np.nan
        else:
            prov = np.nan

        # Region weighted % from counts
        if region_for_compare:
            reg = weighted_percentage(scope[scope["Region"] == region_for_compare])
        else:
            reg = np.nan

        # National weighted % from counts
        nat = weighted_percentage(scope)

        trend_rows.append({
            "Month_dt": mdt,
            "Provider": prov,
            "Region weighted": reg,
            "National weighted": nat,
        })

    trend = pd.DataFrame(trend_rows)
    trend["Month_lbl"] = trend["Month_dt"].dt.strftime("%b-%y")

    # NEW: 1dp labels for hover (keep 2dp rule only if you ever need it later)
    fmt = lambda v: ("—" if pd.isna(v) else f"{v:.1f}%")
    trend["ProvLbl"] = trend["Provider"].map(fmt)
    trend["RegLbl"]  = trend["Region weighted"].map(fmt)
    trend["NatLbl"]  = trend["National weighted"].map(fmt)


    trace_kwargs = dict(mode="lines+markers", cliponaxis=False)

    fig_trend = go.Figure()

    # Provider — solid blue (same as Quarterly)
    if trend["Provider"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend["Month_dt"], y=trend["Provider"],
            name=(provider_code or "Provider"),
            line=dict(width=3, color="#327AD1"),
            customdata=trend["ProvLbl"],
            hovertemplate="<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))

    # Region (weighted) — dashed grey
    if trend["Region weighted"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend["Month_dt"], y=trend["Region weighted"],
            name=f"{region_for_compare or 'Region'} (weighted)",
            line=dict(width=2, dash="dash", color="#6B7280"),
            customdata=trend["RegLbl"],
            hovertemplate="<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))

    # National (weighted) — dotted teal
    if trend["National weighted"].notna().any():
        fig_trend.add_trace(go.Scatter(
            x=trend["Month_dt"], y=trend["National weighted"],
            name="National (weighted)",
            line=dict(width=2, dash="dot", color="#0C988F"),
            customdata=trend["NatLbl"],
            hovertemplate="<b>%{fullData.name}</b>: %{customdata}<extra></extra>",
            **trace_kwargs
        ))


    window_end = latest_dt.strftime("%b-%y") if pd.notnull(latest_dt) else ""
    months_n = len(trend)

    fig_trend.update_layout(
        title=dict(
            text=f"{(provider_code or '').strip()} % Value trend (last {months_n} months, to {window_end})".strip(),
            x=0, xanchor="left"
        ),
        margin=dict(t=56, r=10, l=10, b=0),  # Reduced bottom margin
        legend=dict(orientation="h", y=-0.19, yanchor="top", x=-0.03, xanchor="left", xref="paper"),
        hovermode="x unified",
        template="plotly_white",  # Simple clean template
        height=320
    )

    fig_trend.update_yaxes(
        autorange=True, 
        ticksuffix="%", 
        title_text=None, 
        automargin=True
    )

    fig_trend.update_xaxes(
        tickvals=trend["Month_dt"], 
        ticktext=trend["Month_lbl"], 
        title_text=None
    )
    TREND_H = 450
    fig_trend.update_layout(autosize=False, height=TREND_H)
    st.plotly_chart(fig_trend, use_container_width=True)

with right:
    st.markdown('<div class="rhs-sticky">', unsafe_allow_html=True)

    if not provider_code:
        st.markdown(
            (
                '<div class="metrics-panel cards-plain">'
                '<div class="metrics-panel-title">Metrics within domain</div>'
                '<div class="metric-item"><div class="metric-name">'
                'Select a provider to see ranks across all metrics.'
                '</div></div></div>'
            ),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        scope = df[(df["Month_disp"] == month) & (df["Domain"] == domain)].copy()
        metrics_all = sorted(scope["Metric"].dropna().unique().tolist())

        cards = []
        for m in metrics_all:
            r = scope[(scope["Metric"] == m) & (scope["Provider_Code"] == provider_code)]

            rank_txt = "—"
            reg_txt  = "—"
            pct_txt  = "—"
            pct_val  = 0.0

            if not r.empty:
                rr = r.iloc[0]

                # 1) National rank (left caption)
                rank_txt = "—" if pd.isna(rr["Rank"]) else f"{int(rr['Rank'])}"

                # 2) Regional rank text from existing logic
                if pd.isna(rr.get("Rank_Region")) and pd.isna(rr.get("Region_Size")):
                    reg_txt = "—"
                else:
                    rr_val = "—" if pd.isna(rr.get("Rank_Region")) else f"{int(rr['Rank_Region'])}"
                    if pd.isna(rr.get("Region_Size")):
                        reg_txt = rr_val
                    else:
                        reg_txt = f"{rr_val} out of {int(rr['Region_Size'])}"

                # 3) % display (keep your 1dp / 2dp rule)
                if pd.isna(rr["Percent"]):
                    pct_txt = "—"
                else:
                    pct_txt = f"{rr['Percent']:.2f}%" if str(m).strip().lower() == "52+ weeks" else f"{rr['Percent']:.1f}%"
                    pct_val = float(rr["Percent"])

            # width for the bar (safeguard 0–100)
            width = max(0.0, min(100.0, pct_val))
            # "X out of Y" per metric (national denominator for that metric)
            nat_n_metric = scope[scope["Metric"] == m]["Provider_Code"].nunique()
            nat_label = format_overall_rank_text(rr["Rank"] if not r.empty else pd.NA, nat_n_metric)

            reg_label = f"Regional Rank: {reg_txt}" if reg_txt != "—" else "Regional Rank: —"

            card_html = (
                f'<div class="progress-card">'
                f'  <div class="progress-head">'
                f'    <div class="progress-name">{html.escape(str(m))}</div>'
                f'    <div class="progress-percent">{pct_txt}</div>'
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
            '<div class="metrics-panel-title">Metrics within domain</div>'
            '<div class="progress-list">' + "".join(cards) + '</div>'
            '</div>'
        )
        st.markdown(panel_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # close .rhs-sticky

# Add a thin separator line above the chart
st.markdown('<hr class="chart-separator">', unsafe_allow_html=True)


# 3) Region vs National comparison (weighted)

# Decide which region to use: selected region, else provider's region (if provider chosen)
region_for_compare = region_selected
if region_for_compare is None and provider_code:
    row_any = df_mdm.loc[df_mdm["Provider_Code"] == provider_code]
    if not row_any.empty:
        region_for_compare = row_any.iloc[0]["Region"]

# Scope = all providers in selected Month + Domain + Metric
comp_scope = df_mdm.copy()

# Use the weighted_percentage function for calculations
nat_pct = weighted_percentage(comp_scope)

if region_for_compare:
    reg_scope = comp_scope[comp_scope["Region"] == region_for_compare]
    reg_pct = weighted_percentage(reg_scope)
else:
    reg_pct = float("nan")

# --- Weighted row: 25% | 25% | 50% (info panel), with equal heights
st.markdown("<div class='weights-row'>", unsafe_allow_html=True)

cA, cB, cC = st.columns([1, 1, 3], gap="small")

with cA:
    st.markdown("<div class='weight-card'>", unsafe_allow_html=True)
    with st.container(border=True):
        if region_for_compare:
            st.metric(
                label=f"{region_for_compare} (weighted)",
                value=("—" if pd.isna(reg_pct) else f"{reg_pct:.1f}%"),
                help="Weighted % = Σ Numerator ÷ Σ Denominator × 100",
            )
        else:
            st.metric(label="Region (weighted)", value="—")
    st.markdown("</div>", unsafe_allow_html=True)

with cB:
    st.markdown("<div class='weight-card'>", unsafe_allow_html=True)
    with st.container(border=True):
        st.metric(
            label="National (weighted)",
            value=("—" if pd.isna(nat_pct) else f"{nat_pct:.1f}%"),
            help="Weighted % = Σ Numerator ÷ Σ Denominator × 100",
        )
    st.markdown("</div>", unsafe_allow_html=True)

with cC:
    st.markdown("<div class='weight-card'>", unsafe_allow_html=True)
    
    weight_info_html = f"""
    <div class="weight-panel">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
            <span style="font-size: 1.1rem;">ℹ️</span>
            <strong style="font-size: 0.95rem; color: var(--brand-teal);">Weighted % — how it is calculated</strong>
        </div>
        <p style="margin: 0; font-size: 0.92rem; line-height: 1.4; color: #374151;">
            We use counts (not a simple average of provider percentages). Why weighted? Using counts gives more influence to higher-volume providers, so the overall rate reflects the activity mix.
        </p>
    </div>
    """
    
    st.markdown(weight_info_html, unsafe_allow_html=True)

# ---------------------------- Table -------------------------------
with st.expander("See filtered data as a table and download"):
    table_cols = COLS  # include Notes_Flags as requested
    table_df = df_mdmr[table_cols].copy() if provider_code is None else df_mdmr[table_cols].copy()
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    def to_csv_bytes(d: pd.DataFrame) -> bytes:
        return d.to_csv(index=False).encode("utf-8")

    def to_xlsx_bytes(d: pd.DataFrame) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as w:
            d.to_excel(w, index=False, sheet_name="Monthly")
        return bio.getvalue()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV", data=to_csv_bytes(table_df), file_name="monthly_filtered.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", data=to_xlsx_bytes(table_df),
                           file_name="monthly_filtered.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
