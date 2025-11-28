# pages/3_Compare_Providers.py — Compare Providers (Monthly & Quarterly)
# ---------------------------------------------------------------------
# One page to compare Provider A vs B for either Monthly or Quarterly data.
# - Upload both CSVs on Homepage (already done).
# - Remembers selections separately for each mode: cmp_m_* and cmp_q_*.
# - Reuses your CSS (kpi-card, metrics-panel, progress-card).
# - Header (logo + H1) followed by the blue context banner, matching other pages.
# ---------------------------------------------------------------------

from pathlib import Path
import pandas as pd, numpy as np, io, re, html
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re

# add (or keep) this import in the file header
from decimal import Decimal, ROUND_HALF_UP

# ===================== Page config ======================
st.set_page_config(page_title="NOF Compare Providers", page_icon="🆚", layout="wide")

# Home owns this flag. Pages must not set a default beyond this guard.
st.session_state.setdefault("remember_filters", True)
REMEMBER = st.session_state.get("remember_filters", True)

# ===================== Load shared CSS ==================
def use_ui_css():
    p = Path(__file__).parents[1] / "assets" / "ui.css"
    if p.exists():
        css = p.read_text(encoding="utf-8")
        # tiny cache-buster so browser picks up changes
        css += f"\n/* mtime:{int(p.stat().st_mtime)} */"
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

use_ui_css()

# -------- Empty state for Compare Providers (shows in main pane) --------
def render_compare_empty_state():

    # Context banner
    st.markdown(
        "<div id='context-banner'>Upload a <b>Monthly</b> or <b>Quarterly</b> CSV on the Homepage to start comparing providers.</div>",
        unsafe_allow_html=True,
    )

    # ---------- Line A: add a little space under the banner ----------
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Two columns: left = how to start, right = 1-paragraph description
    left, right = st.columns([0.50, 0.50], gap="medium")

    with left:
        st.markdown(
            """
            <div class="metrics-panel required-cols compact">
              <div class="metrics-panel-title">Get started</div>
              <ol>
                <li>Go to <b>Homepage</b> and upload a <b>Monthly</b> or <b>Quarterly</b> CSV.</li>
                <li>Return here and choose <b>Frequency</b>, pick a <b>Month/Quarter</b>, then <b>Domain</b> and <b>Metric</b>.</li>
                <li>Select <b>Provider A</b> and <b>Provider B</b> in the sidebar.</li>
              </ol>
              <p><em>Tip:</em> You only need one file (Monthly or Quarterly) to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="metrics-panel required-cols compact">
            <div class="metrics-panel-title">What this page does</div>
            <p>
                The Compare Providers dashboard gives a <b><i>clear, side-by-side view</i></b> of <b><i>two trusts</i></b> on the <b><i>same metric and period</i></b>.
                After choosing a frequency and filters, you’ll see <b><i>KPIs for each provider</i></b> (<b><i>overall rank</i></b>, <b><i>regional rank</i></b>, <b><i>% value</i></b>, <b><i>numerator</i></b>, <b><i>denominator</i></b>),
                showing where both sit within the <b><i>national distribution</i></b>, and a <b><i>trend over the last 24 months or 8 quarters</i></b> with <b><i>regional (dashed)</i></b> and <b><i>national (dotted) weighted averages</i></b>.
                Use the <b><i>domain cards</i></b> to quickly scan other measures in the chosen domain and <b><i>spot where the gap is largest</i></b>.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Line B: add a little space below the two-column row ----------
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)


# ========== Title + Logo helpers (shared pattern) =========
ROOT_ASSETS  = Path(__file__).parents[1] / "assets"
PAGES_ASSETS = Path(__file__).parent / "assets"
for candidate in (ROOT_ASSETS / "NOF_Logo.svg", PAGES_ASSETS / "NOF_Logo.svg"):
    if candidate.is_file():
        LOGO_FILE = candidate
        break
else:
    LOGO_FILE = None

def read_svg_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

logo_svg = read_svg_file(LOGO_FILE) if LOGO_FILE else ""

# =================== Map common variants to your canonical names
_NAME_ALIASES = {
    # codes
    "provider code": "Provider_Code",
    "provider_code": "Provider_Code",
    "org code": "Provider_Code",
    "organisation code": "Provider_Code",
    "organization code": "Provider_Code",
    "trust code": "Provider_Code",
    "orgcode": "Provider_Code",

    # names
    "provider name": "Provider_Name",
    "provider_name": "Provider_Name",
    "organisation name": "Provider_Name",
    "organization name": "Provider_Name",
    "org name": "Provider_Name",
    "trust name": "Provider_Name",

    # percent
    "% value": "%_Value",
    "%_value": "%_Value",
    "percent": "%_Value",
    "percentage": "%_Value",

    # ranks / region
    "rank": "Rank",
    "national rank": "Rank",
    "rank national": "Rank",
    "regional rank": "Rank_Region",
    "rank region": "Rank_Region",
    "region size": "Region_Size",
    "regionsize": "Region_Size",

    # quarterly-only extras
    "months covered": "Months_Covered",
    "covered months": "Covered_Months",
}

# Brand colours (use same across charts)
A_COLOR    = "#AE2573"   # NHS Yellow – Provider A
B_COLOR    = "#327AD1"   # NHS Blue   – Provider B
OTHERS_BAR = "#D5DAE1"   # Grey for everyone else
REG_LINE   = "#6B7280"   # Region weighted (dashed)
NAT_LINE   = "#0C988F"   # National weighted (dotted)

def _canonical_key(s: str) -> str:
    """lowercase, collapse to letters/numbers/spaces (no punctuation), compact spaces"""
    s = re.sub(r"[^a-z0-9]+", " ", s.strip().lower())
    return re.sub(r"\s+", " ", s)

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to your canonical names."""
    rename = {}
    for c in df.columns:
        key = _canonical_key(c)
        if key in _NAME_ALIASES:
            rename[c] = _NAME_ALIASES[key]
    return df.rename(columns=rename)


# ===================== Parsers & helpers =================
def _to_float_percent(val):
    """Convert '76.8%' or '0.768' -> 76.8 ; None/'' -> NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return np.nan
    return v * 100.0 if 0 < v <= 1.0 else v

@st.cache_data(show_spinner=False)
def parse_monthly(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    df = standardise_columns(df)

    COLS = ["Month","Domain","Metric","Region","Provider_Code","Provider_Name",
            "Numerator","Denominator","%_Value","Rank","Rank_Region","Region_Size","Data_Date_Used"]
    df = df[[c for c in COLS if c in df.columns]].copy()

    # clean strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Month → dt + label (unchanged)
    dt_us = pd.to_datetime(df["Month"], dayfirst=False, errors="coerce")
    dt_uk = pd.to_datetime(df["Month"], dayfirst=True,  errors="coerce")
    dt = dt_us if dt_us.notna().sum() >= dt_uk.notna().sum() else dt_uk
    df["Month_dt"]   = dt.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1) + pd.offsets.Day(1)
    df["Month_disp"] = df["Month_dt"].dt.strftime("%b-%y")

    # numerics first
    for c in ["Numerator","Denominator","Rank","Rank_Region","Region_Size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

    # robust Percent
    df = ensure_percent(df)
    return df


@st.cache_data(show_spinner=False)
def parse_quarterly(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    df = standardise_columns(df)

    COLS = ["Quarter","Domain","Metric","Region","Provider_Code","Provider_Name",
            "Numerator","Denominator","%_Value","Rank","Rank_Region","Region_Size",
            "Months_Covered","Covered_Months"]
    df = df[[c for c in COLS if c in df.columns]].copy()

    # clean strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # quarter ordering (unchanged)
    order = df["Quarter"].astype("category")
    order = order.cat.set_categories(pd.unique(df["Quarter"]), ordered=True)
    df["Quarter"] = order
    df["Quarter_idx"] = df["Quarter"].cat.codes

    # numerics first
    for c in ["Numerator","Denominator","Rank","Rank_Region","Region_Size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")

    # robust Percent
    df = ensure_percent(df)
    return df


def weighted_percentage(sub: pd.DataFrame) -> float:
    """100 * sum(num)/sum(den); NaN if no denom; clip [0,100]."""
    num = pd.to_numeric(sub.get("Numerator"), errors="coerce").fillna(0).sum()
    den = pd.to_numeric(sub.get("Denominator"), errors="coerce").fillna(0).sum()
    if not den or np.isnan(den):
        return np.nan
    return float(np.clip(100.0 * num / den, 0.0, 100.0))

def provider_labels(sub: pd.DataFrame) -> list[str]:
    # Happy path
    if {"Provider_Code","Provider_Name"}.issubset(sub.columns):
        tmp = sub[["Provider_Code","Provider_Name"]].drop_duplicates().sort_values("Provider_Code")
        return [f"{r.Provider_Code} — {r.Provider_Name}" for _, r in tmp.iterrows()]

    # Fallbacks (should rarely run if standardise_columns worked)
    cols = { _canonical_key(c): c for c in sub.columns }
    code_col = cols.get("provider code") or cols.get("org code") or cols.get("organisation code")
    name_col = cols.get("provider name") or cols.get("organisation name") or cols.get("org name")

    if code_col and name_col:
        tmp = sub[[code_col, name_col]].drop_duplicates().sort_values(code_col)
        return [f"{row[code_col]} — {row[name_col]}" for _, row in tmp.iterrows()]
    if code_col:  # at least show codes
        tmp = sub[[code_col]].drop_duplicates().sort_values(code_col)
        return [f"{row[code_col]} — {row[code_col]}" for _, row in tmp.iterrows()]

    # Nothing identifiable
    return []

def extract_code(lbl: str) -> str | None:
    return None if not lbl or lbl == "(None)" else lbl.split("—")[0].strip()

def remembered(key: str, default):
    if REMEMBER and key in st.session_state:
        return st.session_state[key]
    return default

def name_then_code(label: str) -> str:
    """Convert 'CODE — Name' -> 'Name (CODE)' for headings."""
    if not label or label == "(None)":
        return label
    parts = label.split("—", 1)
    if len(parts) == 2:
        code = parts[0].strip()
        name = parts[1].strip()
        return f"{name} ({code})"
    return label

def name_code_with_region(label: str, lookup_df: pd.DataFrame) -> str:
    """
    Turn 'CODE — Name' into 'Name (CODE — Region)'.
    Falls back to 'Name (CODE)' if Region isn't found.
    """
    if not label or label == "(None)":
        return label
    parts = label.split("—", 1)
    if len(parts) == 2:
        code = parts[0].strip()
        name = parts[1].strip()
    else:
        # Unexpected label format; keep as-is
        return label

    region_txt = None
    if "Region" in lookup_df.columns and "Provider_Code" in lookup_df.columns:
        vals = (lookup_df.loc[lookup_df["Provider_Code"] == code, "Region"]
                .dropna().astype(str).unique().tolist())
        if vals:
            region_txt = vals[0]

    return f"{name} ({code})" if region_txt else f"{name} ({code})"


def safe_index(options: list, wanted, default_idx: int = 0) -> int:
    """Return a safe index into options. If wanted not present, return default_idx (clamped)."""
    if not options:
        return 0
    if wanted in options:
        return options.index(wanted)
    return max(0, min(default_idx, len(options) - 1))

def with_previous(labels: list[str], *prevs: str) -> list[str]:
    """Build select options: ['(None)'] + unique(labels) + any previous selections (if missing)."""
    out = ["(None)"]
    seen = set(out)
    for lbl in labels:
        if lbl and lbl not in seen:
            out.append(lbl); seen.add(lbl)
    for p in prevs:
        if p and p not in ("(None)",) and p not in seen:
            out.append(p); seen.add(p)
    return out

def ensure_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 0–100 Percent column for charts/cards."""
    # Try a provided % column
    if "%_Value" in df.columns and df["%_Value"].notna().any():
        p = df["%_Value"].apply(_to_float_percent)
    else:
        # Compute from Numerator/Denominator
        num = pd.to_numeric(df.get("Numerator"), errors="coerce")
        den = pd.to_numeric(df.get("Denominator"), errors="coerce")
        p = (num / den) * 100

    # If it still looks like a 0–1 scale, normalise to 0–100
    med = p.dropna().median()
    if pd.notna(med) and med <= 5:   # e.g., 0.72 → 72
        p = p * 100

    # Clean up: finite, clipped to [0,100]
    p = p.where(np.isfinite(p), np.nan).clip(lower=0, upper=100)
    df["Percent"] = p
    return df

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


def get_region_for_provider(code: str | None, lookup_df: pd.DataFrame) -> str | None:
    """Return the first Region for a provider code from the lookup_df (scope0)."""
    if not code or "Region" not in lookup_df.columns or "Provider_Code" not in lookup_df.columns:
        return None
    vals = (lookup_df.loc[lookup_df["Provider_Code"] == code, "Region"]
            .dropna().astype(str).unique().tolist())
    return vals[0] if vals else None

# ===================== Helper function for dynamic summary =====================
def generate_summary(
    scope: pd.DataFrame,
    scope0: pd.DataFrame,
    df_full: pd.DataFrame,
    provA: str | None,
    provB: str | None,
    labelA: str,
    labelB: str,
    metric: str,
    domain: str,
    period: str,  # month or quarter
    region_selected: str | None,
    mode: str
) -> str:
    """Generate a 2-paragraph dynamic summary based on current filters."""
    
    # Helper to extract provider name from label
    def get_name(label: str) -> str:
        if not label or label == "(None)":
            return ""
        parts = label.split("—", 1)
        return parts[1].strip() if len(parts) == 2 else label
    
    # Helper to get provider data
    def get_prov_data(code: str | None):
        if not code:
            return None
        r = scope[scope["Provider_Code"] == code]
        return r.iloc[0] if not r.empty else None
    
    nameA = get_name(labelA)
    nameB = get_name(labelB)
    dataA = get_prov_data(provA)
    dataB = get_prov_data(provB)
    
    # Calculate regional and national weighted averages
    reg_avg = nat_avg = None
    if region_selected:
        reg_scope = scope0[scope0["Region"] == region_selected]
        reg_num = pd.to_numeric(reg_scope.get("Numerator"), errors="coerce").sum()
        reg_den = pd.to_numeric(reg_scope.get("Denominator"), errors="coerce").sum()
        if reg_den > 0:
            reg_avg = (reg_num / reg_den) * 100
    
    nat_num = pd.to_numeric(scope0.get("Numerator"), errors="coerce").sum()
    nat_den = pd.to_numeric(scope0.get("Denominator"), errors="coerce").sum()
    if nat_den > 0:
        nat_avg = (nat_num / nat_den) * 100
    
    # Format helper
    def fmt_pct(val, met):
        if pd.isna(val):
            return "—"
        return f"{val:.2f}%" if str(met).strip().lower() == "52+ weeks" else f"{val:.1f}%"
    
    # Paragraph 1: Current period comparison
    if dataA is not None and dataB is not None:
        pctA = dataA.get("Percent", np.nan)
        pctB = dataB.get("Percent", np.nan)
        rankA = int(dataA["Rank"]) if pd.notna(dataA.get("Rank")) else None
        rankB = int(dataB["Rank"]) if pd.notna(dataB.get("Rank")) else None
        
        nat_count = scope0["Provider_Code"].nunique()
        reg_sizeA = int(dataA["Region_Size"]) if pd.notna(dataA.get("Region_Size")) else None
        reg_rankA = int(dataA["Rank_Region"]) if pd.notna(dataA.get("Rank_Region")) else None
        reg_sizeB = int(dataB["Region_Size"]) if pd.notna(dataB.get("Region_Size")) else None
        reg_rankB = int(dataB["Rank_Region"]) if pd.notna(dataB.get("Rank_Region")) else None
        
        region_txt = f" within the <b>{html.escape(region_selected)}</b> region" if region_selected else " across all regions"
        
        p1 = (
            f"<p>In <b>{html.escape(period)}</b>, <b>{html.escape(nameA)} ({html.escape(provA)})</b> achieved a <b>{fmt_pct(pctA, metric)}</b> "
            f"performance rate for <b>{html.escape(domain)} → {html.escape(metric)}</b>{region_txt}, "
            f"securing an overall national rank of <b>{rankA}</b> out of {nat_count} trusts"
        )
        if reg_rankA and reg_sizeA:
            p1 += f" and a regional rank of <b>{reg_rankA}</b> out of {reg_sizeA} trusts"
        
        p1 += f". By comparison, <b>{html.escape(nameB)} ({html.escape(provB)})</b> recorded <b>{fmt_pct(pctB, metric)}</b>, "
        p1 += f"ranking <b>{rankB}</b> nationally"
        
        if reg_rankB and reg_sizeB:
            p1 += f" and <b>{reg_rankB}</b> regionally"
        
        if pd.notna(pctA) and pd.notna(pctB):
            gap = abs(pctB - pctA)
            leader = nameB if pctB > pctA else nameA
            p1 += f". This represents a <b>{gap:.1f} percentage point</b> gap, with {html.escape(leader)} demonstrating stronger performance"
        p1 += "."
        
        # Add weighted averages comparison
        if reg_avg and pd.notna(reg_avg):
            p1 += f" The regional weighted average for this metric stood at <b>{fmt_pct(reg_avg, metric)}</b>"
        if nat_avg and pd.notna(nat_avg):
            if reg_avg and pd.notna(reg_avg):
                p1 += f", while the national weighted average was <b>{fmt_pct(nat_avg, metric)}</b>"
            else:
                p1 += f" The national weighted average for this metric stood at <b>{fmt_pct(nat_avg, metric)}</b>"
        p1 += ".</p>"
        
    elif dataA is not None:
        pctA = dataA.get("Percent", np.nan)
        rankA = int(dataA["Rank"]) if pd.notna(dataA.get("Rank")) else None
        nat_count = scope0["Provider_Code"].nunique()
        
        p1 = (
            f"<p>In <b>{html.escape(period)}</b>, <b>{html.escape(nameA)} ({html.escape(provA)})</b> achieved a <b>{fmt_pct(pctA, metric)}</b> "
            f"performance rate for <b>{html.escape(domain)} → {html.escape(metric)}</b>, ranking <b>{rankA}</b> out of {nat_count} trusts. "
            f"<b>Provider B</b> has not been selected for comparison.</p>"
        )
    else:
        p1 = f"<p>No provider data available for the selected filters in <b>{html.escape(period)}</b>.</p>"
    
    # Paragraph 2: Trend analysis
    period_col = "Month_dt" if mode == "Monthly" else "Quarter"
    lookback = 24 if mode == "Monthly" else 8
    period_label = "months" if mode == "Monthly" else "quarters"
    
    # Get historical data
    hist = df_full[(df_full["Domain"] == domain) & (df_full["Metric"] == metric)].copy()
    if mode == "Monthly":
        hist = hist.sort_values("Month_dt")
        periods = hist["Month_dt"].drop_duplicates().tail(lookback).tolist()
    else:
        hist = hist.sort_values("Quarter_idx")
        periods = hist["Quarter"].drop_duplicates().tail(lookback).tolist()
    
    p2_parts = []
    
    if provA and dataA is not None:
        histA = hist[hist["Provider_Code"] == provA]
        if not histA.empty and len(histA) > 1:
            pcts = histA["Percent"].dropna()
            if len(pcts) > 0:
                min_pct = pcts.min()
                max_pct = pcts.max()
                p2_parts.append(
                    f"<b>{html.escape(nameA)} ({html.escape(provA)})</b> has shown performance ranging from "
                    f"<b>{fmt_pct(min_pct, metric)}</b> to <b>{fmt_pct(max_pct, metric)}</b> over the preceding {lookback} {period_label}"
                )
    
    if provB and dataB is not None:
        histB = hist[hist["Provider_Code"] == provB]
        if not histB.empty and len(histB) > 1:
            pcts = histB["Percent"].dropna()
            if len(pcts) > 0:
                min_pct = pcts.min()
                max_pct = pcts.max()
                p2_parts.append(
                    f"<b>{html.escape(nameB)} ({html.escape(provB)})</b> has demonstrated performance between "
                    f"<b>{fmt_pct(min_pct, metric)}</b> and <b>{fmt_pct(max_pct, metric)}</b> over the same period"
                )
    
    if p2_parts:
        p2 = f"<p>{'. '.join(p2_parts)}. The trend chart displays their performance trajectories (solid lines) alongside regional (dashed grey) and national (dotted teal) weighted averages, providing context for relative performance over time.</p>"
    else:
        p2 = f"<p>The trend chart displays performance over the last {lookback} {period_label}, comparing regional (dashed grey) and national (dotted teal) weighted averages.</p>"
    
    return p1 + p2

# ===================== Sidebar: frequency & files ================
with st.sidebar:
    st.header("📁 Data in use")

    modes = []
    if "monthly_bytes" in st.session_state:
        modes.append("Monthly")
    if "quarterly_bytes" in st.session_state:
        modes.append("Quarterly")

    has_data = bool(modes)

    # --- EMPTY STATE: show ONLY a warning + CTA, nothing else ---
    if not has_data:
        st.warning("No files loaded. Upload on Homepage first.", icon="⚠️")
        if st.button("Go to Homepage", key="cmp_goto_home"):
            st.switch_page("Homepage.py")
    # --- DATA PRESENT: show the real controls ---
    else:
        default_mode = "Monthly" if "monthly_bytes" in st.session_state else "Quarterly"
        wanted_mode = remembered("cmp_mode", default_mode)
        idx = safe_index(modes, wanted_mode, default_idx=0)

        mode = st.radio("Frequency", options=modes, index=idx, horizontal=True)
        if REMEMBER:
            st.session_state["cmp_mode"] = mode

        # Show current file + clear (guarded so they only appear when that file exists)
        if mode == "Monthly":
            if "monthly_name" in st.session_state:
                st.caption(f"Using: {st.session_state['monthly_name']}")
            if "monthly_bytes" in st.session_state and st.button("Clear Monthly", key="cmp_clear_m"):
                st.session_state.pop("monthly_bytes", None)
                st.session_state.pop("monthly_name",  None)
                st.rerun()
        else:  # Quarterly
            if "quarterly_name" in st.session_state:
                st.caption(f"Using: {st.session_state['quarterly_name']}")
            if "quarterly_bytes" in st.session_state and st.button("Clear Quarterly", key="cmp_clear_q"):
                st.session_state.pop("quarterly_bytes", None)
                st.session_state.pop("quarterly_name",  None)
                st.rerun()


# If no Monthly/Quarterly file is available, show the main-pane empty state and stop once.
if not has_data:
    render_compare_empty_state()
    st.stop()


# ===================== Load data frame per mode ===================
if mode == "Monthly":
    df = parse_monthly(st.session_state["monthly_bytes"])
else:
    df = parse_quarterly(st.session_state["quarterly_bytes"])

# ===================== Filters (per mode) =========================
DOMAIN_ORDER = {"A&E": 0, "Cancer": 1, "RTT": 2, "Diagnostic": 3}

if mode == "Monthly":
    # Month
    months = df.sort_values("Month_dt")["Month_disp"].unique().tolist()
    m_def  = months[-1]
    month  = st.sidebar.selectbox(
        "Month", months, 
        index=safe_index(months, remembered("cmp_m_month", m_def), default_idx=len(months)-1)
    )

    df_scoped = df[df["Month_disp"] == month]

    # Domain
    domains = sorted(df_scoped["Domain"].dropna().unique().tolist(), key=lambda d: DOMAIN_ORDER.get(d, 999))
    domain  = st.sidebar.selectbox(
        "Domain", domains, 
        index=safe_index(domains, remembered("cmp_m_domain", domains[0]))
    )

    # Metric
    metrics = sorted(df_scoped[df_scoped["Domain"] == domain]["Metric"].dropna().unique().tolist())
    if not metrics:
        st.warning("No metrics found for this domain in the selected month.")
        st.stop()

    metric  = st.sidebar.selectbox(
        "Metric", metrics, 
        index=safe_index(metrics, remembered("cmp_m_metric", metrics[0]))
    )

    # Region
    scope0  = df_scoped[(df_scoped["Domain"] == domain) & (df_scoped["Metric"] == metric)].copy()
    nat_total = scope0["Provider_Code"].nunique()
    regions = ["(All Regions)"] + sorted(scope0["Region"].dropna().unique().tolist())
    region_choice = st.sidebar.selectbox(
        "Region", regions, 
        index=safe_index(regions, remembered("cmp_m_region", "(All Regions)"))
    )

    region_selected = None if region_choice == "(All Regions)" else region_choice

    scope = scope0 if region_selected is None else scope0[scope0["Region"] == region_selected]
    
    # Domain-wide rows (for the right-hand cards) — NOT filtered by metric
    domain_rows0 = df_scoped[df_scoped["Domain"] == domain].copy()  # unfiltered by region
    domain_rows_cards = domain_rows0 if region_selected is None else domain_rows0[domain_rows0["Region"] == region_selected]
    
    labels = provider_labels(scope)

    # Defaults from page memory
    prevL = remembered("cmp_m_left",  labels[0] if labels else "(None)")
    prevR = remembered("cmp_m_right", labels[1] if len(labels) > 1 else "(None)")

    # NEW: if a shared provider exists, prefer it for Provider A
    shared_code = st.session_state.get("shared_provider_code") if REMEMBER else None
    if shared_code:
        shared_label = next((lbl for lbl in labels if extract_code(lbl) == shared_code), None)
        if shared_label:
            prevL = shared_label

    # Build options bringing forward any previous picks
    prov_options = with_previous(labels, prevL, prevR)

    colL = st.sidebar.selectbox(
        "Provider A",
        prov_options,
        index=safe_index(prov_options, prevL)
    )
    colR = st.sidebar.selectbox(
        "Provider B",
        prov_options,
        index=safe_index(prov_options, prevR)
    )

    provA = extract_code(colL)
    provB = extract_code(colR)

    if REMEMBER:
        st.session_state.update({
            "cmp_m_month": month, "cmp_m_domain": domain, "cmp_m_metric": metric,
            "cmp_m_region": region_choice, "cmp_m_left": colL, "cmp_m_right": colR
        })

else:
    # Quarterly mode
    quarters = df.sort_values("Quarter_idx")["Quarter"].astype(str).unique().tolist()
    q_def    = quarters[-1]
    quarter  = st.sidebar.selectbox(
        "Quarter", quarters, 
        index=safe_index(quarters, remembered("cmp_q_quarter", q_def), default_idx=len(quarters)-1)
    )

    df_scoped = df[df["Quarter"].astype(str) == quarter]

    # Domain
    domains = sorted(df_scoped["Domain"].dropna().unique().tolist(), key=lambda d: DOMAIN_ORDER.get(d, 999))
    domain  = st.sidebar.selectbox(
        "Domain", domains, 
        index=safe_index(domains, remembered("cmp_q_domain", domains[0]))
    )

    # Metric
    metrics = sorted(df_scoped[df_scoped["Domain"] == domain]["Metric"].dropna().unique().tolist())
    if not metrics:
        st.warning("No metrics found for this domain in the selected quarter.")
        st.stop()

    metric  = st.sidebar.selectbox(
        "Metric", metrics, 
        index=safe_index(metrics, remembered("cmp_q_metric", metrics[0]))
    )

    # Region
    scope0  = df_scoped[(df_scoped["Domain"] == domain) & (df_scoped["Metric"] == metric)].copy()
    nat_total = scope0["Provider_Code"].nunique()

    regions = ["(All Regions)"] + sorted(scope0["Region"].dropna().unique().tolist())
    region_choice = st.sidebar.selectbox(
        "Region", regions, 
        index=safe_index(regions, remembered("cmp_q_region", "(All Regions)"))
    )

    region_selected = None if region_choice == "(All Regions)" else region_choice

    scope = scope0 if region_selected is None else scope0[scope0["Region"] == region_selected]
    
    # Domain-wide rows (for the right-hand cards) — NOT filtered by metric
    domain_rows0 = df_scoped[df_scoped["Domain"] == domain].copy()
    domain_rows_cards = domain_rows0 if region_selected is None else domain_rows0[domain_rows0["Region"] == region_selected]

    labels = provider_labels(scope)

    prevL = remembered("cmp_q_left",  labels[0] if labels else "(None)")
    prevR = remembered("cmp_q_right", labels[1] if len(labels) > 1 else "(None)")
    
    # NEW: if a shared provider exists, prefer it for Provider A (Quarterly mode)
    shared_code = st.session_state.get("shared_provider_code") if REMEMBER else None
    if shared_code:
        shared_label = next((lbl for lbl in labels if extract_code(lbl) == shared_code), None)
        if shared_label:
            prevL = shared_label
    
    prov_options = with_previous(labels, prevL, prevR)

    colL = st.sidebar.selectbox(
        "Provider A",
        prov_options,
        index=safe_index(prov_options, prevL)
    )
    colR = st.sidebar.selectbox(
        "Provider B",
        prov_options,
        index=safe_index(prov_options, prevR)
    )

    provA = extract_code(colL)
    provB = extract_code(colR)

    if REMEMBER:
        st.session_state.update({
            "cmp_q_quarter": quarter, "cmp_q_domain": domain, "cmp_q_metric": metric,
            "cmp_q_region": region_choice, "cmp_q_left": colL, "cmp_q_right": colR
        })

# Keep cross-page shared provider behaviour (A takes precedence when chosen)
if REMEMBER and provA:
    st.session_state["shared_provider_code"] = provA

# ===================== Header & Context banner ====================
if mode == "Monthly":
    context_html = (
        f"Comparing <b>{html.escape(domain)}</b> → <b>{html.escape(metric)}</b> in <b>{html.escape(month)}</b>"
        + (f" for <b>{html.escape(region_selected)}</b>" if region_selected else " across <b>all regions</b>")
    )
else:
    context_html = (
        f"Comparing <b>{html.escape(domain)}</b> → <b>{html.escape(metric)}</b> in <b>{html.escape(quarter)}</b>"
        + (f" for <b>{html.escape(region_selected)}</b>" if region_selected else " across <b>all regions</b>")
    )

# ------------------------------------------------------------------

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# --------------------------- Header -------------------------------
st.markdown(
    f"""
    <div class="app-header" role="banner" aria-label="Header">
      <div class="app-logo" aria-hidden="true">{logo_svg}</div>
      <h1 id="page-title">NOF Compare Providers</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===================== Context banner ======================
st.markdown(f"<div id='context-banner'>{context_html}</div>", unsafe_allow_html=True)


# ===================== KPI helpers =========================
def kpi_values(df_src: pd.DataFrame, code: str, metric_name: str):
    r = df_src.loc[df_src["Provider_Code"] == code]
    if r.empty:
        return None
    x = r.iloc[0]
    rank   = "—" if pd.isna(x.get("Rank")) else f"{int(x['Rank'])}"
    rreg   = "—" if pd.isna(x.get("Rank_Region")) else f"{int(x['Rank_Region'])}"
    rsize  = ""  if pd.isna(x.get("Region_Size")) else f"<span class='muted'> / {int(x['Region_Size'])}</span>"
    pct    = x.get("Percent", np.nan)
    pct_ht = "—" if pd.isna(pct) else (f"{pct:.2f}%" if str(metric_name).strip().lower() == "52+ weeks" else f"{pct:.1f}%")
    num    = "—" if pd.isna(x.get("Numerator")) else f"{int(float(x['Numerator'])):,}"
    den    = "—" if pd.isna(x.get("Denominator")) else f"{int(float(x['Denominator'])):,}"
    return rank, f"{rreg}{rsize}", pct_ht, num, den

def kpi_card_html(title: str, value_html: str) -> str:
    return f"<div class='kpi-card'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value_html}</div></div>"

def render_provider_kpis(
    scope_df: pd.DataFrame,
    provider_code: str | None,
    label_html: str,
    metric_name: str,
    heading_color: str = "#111827",
    region_line: str | None = None,
    nat_total: int | None = None
):

    if not provider_code:
        st.info(f"Pick **{label_html.split(':',1)[0]}** in the sidebar.")
        return

    vals = kpi_values(scope_df, provider_code, metric_name)
    if not vals:
        st.info("No row for that provider in the current filters.")
        return

    rr, rreg, pct, num, den = vals
    overall_html = rr if (rr == "—" or not nat_total) else f"{rr}<span class='muted'> / {int(nat_total)}</span>"
    region_html = f"<div class='prov-kpi-sub'>{html.escape(region_line)}</div>" if region_line else "<div class='prov-kpi-sub'>&nbsp;</div>"
    html_block = f"""
    <div class="prov-block">
      <div class="prov-title-wrap">
        <h3 class="prov-kpi-title" style="color:{heading_color};">{label_html}</h3>
        {region_html}
      </div>
      <div class="kpi-grid">
        {kpi_card_html("Overall Rank", overall_html)}
        {kpi_card_html("Region Rank",  rreg)}
        {kpi_card_html("% Value",      pct)}
        {kpi_card_html("Numerator",    num)}
        {kpi_card_html("Denominator",  den)}
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


# ===================== KPI row (A left, B right) ==================
# First line (coloured): A/B + Name (CODE)
titleA = f"A: {html.escape(name_then_code(colL))}"
titleB = f"B: {html.escape(name_then_code(colR))}"

# Second line: each provider’s own Region from scope0
regionA = get_region_for_provider(provA, scope0)
regionB = get_region_for_provider(provB, scope0)


colA, colB = st.columns(2, gap="large")
with colA:
    render_provider_kpis(scope, provA, titleA, metric, A_COLOR, regionA, nat_total)

with colB:
    render_provider_kpis(scope, provB, titleB, metric, B_COLOR, regionB, nat_total)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ===================== Distribution (current period) ===============
bars = scope0 if region_selected is None else scope0[scope0["Region"] == region_selected]
bars = bars.dropna(subset=["Percent","Provider_Code"]).copy()
bars = bars.sort_values(["Percent", "Provider_Name"], ascending=[False, True])
bars["IsA"] = bars["Provider_Code"].eq(provA) if provA else False
bars["IsB"] = bars["Provider_Code"].eq(provB) if provB else False


dist_title = "Distribution — selected " + ("month" if mode == "Monthly" else "quarter")
st.markdown(f"##### {html.escape(str(metric))}  {dist_title}")   # << same size as the trend title

# Pre-format labels for hover to match Quarterly
bars["PercentLabel"]     = bars.apply(lambda r: format_percent_display(r["Percent"], r["Metric"]), axis=1)
bars["NumLabel"]         = bars["Numerator"].map(format_whole_round)
bars["DenLabel"]         = bars["Denominator"].map(format_whole_round)
bars["RegionRankLabel"]  = bars.apply(lambda r: format_region_rank(r["Rank_Region"], r["Region_Size"]), axis=1)

pxfig = px.bar(
    bars, x="Provider_Code", y="Percent",
    title=None
)

pxfig.update_traces(
    marker_color=np.where(bars["IsA"], A_COLOR,
                   np.where(bars["IsB"], B_COLOR, OTHERS_BAR)),
    customdata=bars[[
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

pxfig.update_layout(
    template="simple_white",
    xaxis_title="Providers",
    yaxis_title=None,
    yaxis_ticksuffix="%",
    xaxis_showticklabels=False,
    height=410,
    margin=dict(l=10, r=10, t=6, b=10)
)
# remove x-axis baseline to match your bar style elsewhere
pxfig.update_xaxes(showline=False, linewidth=0, linecolor="rgba(0,0,0,0)", showgrid=False, zeroline=False, ticks="", mirror=False)


st.plotly_chart(pxfig, use_container_width=True)

# ========== Domain-wide "metrics within domain" cards (tabs) ======
# AFTER (adds space above the line and a tiny space below)
st.markdown("<hr class='chart-separator cmp-sep'>", unsafe_allow_html=True)

def render_progress_cards(
    all_rows: pd.DataFrame,
    provider_code: str | None,
    panel_title: str,
    only_metric: str | None = None,
    nat_frame: pd.DataFrame | None = None,   # << new
):
    if nat_frame is None:
        nat_frame = all_rows  # fallback if caller doesn't supply one
        
    if not provider_code:
        st.markdown(
            '<div class="metrics-panel cards-plain">'
            f'<div class="metrics-panel-title">{html.escape(panel_title)}</div>'
            '<div class="metric-item"><div class="metric-name">Pick a provider above.</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    # ↓ Only the chosen metric, or all if not provided
    if only_metric:
        metrics_all = [only_metric]
    else:
        metrics_all = sorted(all_rows["Metric"].dropna().unique().tolist())

    cards = []
    for m in metrics_all:
        r = all_rows[(all_rows["Metric"] == m) & (all_rows["Provider_Code"] == provider_code)]
        rank_txt = "—"; reg_txt = "—"; pct_txt = "—"; pct_val = 0.0
        if not r.empty:
            rr = r.iloc[0]
            rank_txt = "—" if pd.isna(rr.get("Rank")) else f"{int(rr['Rank'])}"
            if pd.isna(rr.get("Rank_Region")) and pd.isna(rr.get("Region_Size")):
                reg_txt = "—"
            else:
                rr_val = "—" if pd.isna(rr.get("Rank_Region")) else f"{int(rr['Rank_Region'])}"
                reg_txt = rr_val if pd.isna(rr.get("Region_Size")) else f"{rr_val} out of {int(rr['Region_Size'])}"
            if pd.isna(rr.get("Percent")):
                pct_txt = "—"
            else:
                pct_txt = f"{rr['Percent']:.2f}%" if str(m).strip().lower() == "52+ weeks" else f"{rr['Percent']:.1f}%"
                pct_val = float(rr["Percent"])

        width = max(0.0, min(100.0, pct_val))
        # national provider count for THIS metric (ignore region)
        nat_n_metric = nat_frame[nat_frame["Metric"] == m]["Provider_Code"].nunique()

        # captions
        nat_label = "—" if (rank_txt == "—" or nat_n_metric == 0) else f"{rank_txt} out of {int(nat_n_metric)}"
        reg_label = f"Regional Rank: {reg_txt}" if reg_txt != "—" else "Regional Rank: —"

        cards.append(
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

    panel_html = (
        '<div class="metrics-panel cards-plain">'
        f'<div class="metrics-panel-title">{html.escape(panel_title)}</div>'
        '<div class="progress-list">' + "".join(cards) + '</div>'
        '</div>'
    )
    st.markdown(panel_html, unsafe_allow_html=True)

# ===================== Trend (left) + Metrics panel (right) =====================
# st.markdown("<hr class='chart-separator cmp-sep'>", unsafe_allow_html=True)

left, right = st.columns([0.60, 0.40], gap="medium")   # trend wider, cards narrower

with left:
    # ---- Title for the trend chart
    st.markdown(f"##### Trend — {html.escape(str(metric))} ({'last 24 months' if mode=='Monthly' else 'last 8 quarters'})")

    fig = go.Figure()

    if mode == "Monthly":
        dm = df[(df["Domain"] == domain) & (df["Metric"] == metric)].copy()
        dm = dm.sort_values("Month_dt").drop_duplicates(subset=["Month_dt","Provider_Code"], keep="last")
        latest = dm["Month_dt"].max()
        months24 = dm[["Month_dt"]].drop_duplicates().sort_values("Month_dt")["Month_dt"]
        months24 = months24[months24 <= latest].tail(24).tolist()

        def add_series(code, label, color):
            s = dm[dm["Provider_Code"] == code]
            if not s.empty:
                fig.add_trace(go.Scatter(
                    x=s["Month_dt"], y=s["Percent"], name=label,
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(color=color, size=6, line=dict(width=1, color="#FFFFFF"))
                ))

        if provA: add_series(provA, f"A · {provA}", A_COLOR)
        if provB: add_series(provB, f"B · {provB}", B_COLOR)

        def add_weighted(frame, name, dash):
            rows = []
            for mdt in months24:
                scope_m = frame[frame["Month_dt"] == mdt]
                rows.append(dict(x=mdt, y=weighted_percentage(scope_m)))
            s = pd.DataFrame(rows)
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["y"], name=name, mode="lines",
                line=dict(width=2, dash=dash, color=REG_LINE if dash=="dash" else NAT_LINE)
            ))

        if region_selected:
            add_weighted(dm[dm["Region"] == region_selected], f"{region_selected} (weighted)", "dash")
        add_weighted(dm, "National (weighted)", "dot")

    else:
        dq = df[(df["Domain"] == domain) & (df["Metric"] == metric)].copy()
        q_uni = dq[["Quarter","Quarter_idx"]].drop_duplicates().sort_values("Quarter_idx")
        last8 = q_uni.tail(8)["Quarter"].astype(str).tolist()

        def add_series_q(code, label, color):
            s = dq[dq["Provider_Code"] == code].copy()
            if s.empty: return
            s = s.drop_duplicates(subset=["Quarter","Provider_Code"], keep="last").sort_values("Quarter_idx")
            s = s[s["Quarter"].astype(str).isin(last8)]
            fig.add_trace(go.Scatter(
                x=s["Quarter"].astype(str), y=s["Percent"], name=label,
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(color=color, size=6, line=dict(width=1, color="#FFFFFF"))
            ))

        if provA: add_series_q(provA, f"A · {provA}", A_COLOR)
        if provB: add_series_q(provB, f"B · {provB}", B_COLOR)

        def add_weighted_q(frame, name, dash):
            rows = []
            q_u = frame[["Quarter","Quarter_idx"]].drop_duplicates().sort_values("Quarter_idx")
            q_u = q_u[q_u["Quarter"].astype(str).isin(last8)]
            for qlbl in q_u["Quarter"].astype(str):
                rows.append(dict(x=qlbl, y=weighted_percentage(frame[frame["Quarter"].astype(str) == qlbl])))
            s = pd.DataFrame(rows)
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["y"], name=name, mode="lines",
                line=dict(width=2, dash=dash, color=REG_LINE if dash=="dash" else NAT_LINE)
            ))

        if region_selected:
            add_weighted_q(dq[dq["Region"] == region_selected], f"{region_selected} (weighted)", "dash")
        add_weighted_q(dq, "National (weighted)", "dot")

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=6, b=0),       # small top margin; title is outside
        legend=dict(orientation="h", y=-0.15, x=-0.05),
        hovermode="x unified"
    )
    TREND_H = 450
    
    fig.update_layout(autosize=False, height=TREND_H)
    fig.update_yaxes(ticksuffix="%", title=None)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, use_container_width=True)


with right:
    tabA, tabB = st.tabs(["Provider A", "Provider B"])
    with tabA:
        render_progress_cards(domain_rows_cards, provA,
            f"Metrics within domain — {provA} ({mode})",
            nat_frame=domain_rows0)
    with tabB:
        render_progress_cards(domain_rows_cards, provB,
            f"Metrics within domain — {provB} ({mode})",
            nat_frame=domain_rows0)


# ===================== Dynamic Summary Section =====================
# st.markdown("<hr class='chart-separator'>", unsafe_allow_html=True)

# Generate and display the summary
summary_html = generate_summary(
    scope=scope,
    scope0=scope0,
    df_full=df,
    provA=provA,
    provB=provB,
    labelA=colL,
    labelB=colR,
    metric=metric,
    domain=domain,
    period=(month if mode == "Monthly" else quarter),
    region_selected=region_selected,
    mode=mode
)

summary_panel = f"""
<div class="metrics-panel summary-panel compare-summary-panel">
    <div class="summary-title">Performance Analysis Summary</div>
    {summary_html}
</div>
"""

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
st.markdown(summary_panel, unsafe_allow_html=True)