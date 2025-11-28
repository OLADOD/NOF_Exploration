# pages/4_Foundation_Group.py – NOF Foundation Group (FIXED VERSION)
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import html  # ← add this
from textwrap import dedent
import streamlit.components.v1 as components  # optional


# ---------------- Page config ----------------
st.set_page_config(
    page_title="NOF Foundation Group",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("remember_filters", True)

# --------------- Shared CSS ---------------
def _read_text_safely(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return path.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _inject_css_once():
    for p in [Path(__file__).with_name("ui.css"),
              Path(__file__).parent / "assets" / "ui.css",
              Path("ui.css"), Path("assets/ui.css")]:
        if p.exists():
            css = _read_text_safely(p)
            css += f"\n/* mtime:{int(p.stat().st_mtime)} */"
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            break

_inject_css_once()

# ========== Logo helper (matching other pages) =========
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

# ----------------- Column normalisation & formatting -----------------
RENAME = {
    "provider code": "provider_code", "provider_code": "provider_code",
    "provider name": "provider_name", "provider_name": "provider_name",
    "% value": "percent", "%_value": "percent", "percent": "percent",
    "percentage": "percent", "percent_value": "percent",
    "numerator": "numerator", "denominator": "denominator",
    "rank": "rank", "rank_region": "rank_region", "region_size": "region_size",
    "region": "region", "domain": "domain", "metric": "metric",
    "month": "month", "quarter": "quarter"
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    return df

def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if not raw: return pd.DataFrame()
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, engine="python")
    df = _norm_cols(df)

    for c in ("percent", "numerator", "denominator", "rank", "rank_region", "region_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    if "percent" not in df.columns:
        if {"numerator", "denominator"}.issubset(df.columns):
            n = pd.to_numeric(df["numerator"], errors="coerce")
            d = pd.to_numeric(df["denominator"], errors="coerce")
            df["percent"] = np.where(d > 0, (n / d) * 100.0, np.nan)
        else:
            df["percent"] = np.nan

    if "percent" in df.columns:
        df["percent"] = ensure_percent_0_100(df["percent"])
    
    for c in ("provider_code","provider_name","region","domain","metric","month","quarter"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df

def _codes_and_names(df: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in ("provider_code", "provider_name") if c in df.columns]
    if len(need) < 2:
        return pd.DataFrame(columns=["provider_code", "provider_name"])

    tmp = df[["provider_code", "provider_name"]].dropna(subset=["provider_code"]).copy()
    tmp["provider_code"] = tmp["provider_code"].astype(str).str.strip()
    tmp["provider_name"] = tmp["provider_name"].fillna("").astype(str).str.strip()

    tmp["all_caps"] = ~tmp["provider_name"].str.contains(r"[a-z]")
    tmp["name_len"] = tmp["provider_name"].str.len()

    tmp = tmp.sort_values(
        by=["provider_code", "all_caps", "name_len", "provider_name"],
        ascending=[True, True, False, True],
        kind="stable",
    )

    out = tmp.drop_duplicates(subset=["provider_code"], keep="first")[["provider_code", "provider_name"]]
    return out.sort_values(["provider_code", "provider_name"]).reset_index(drop=True)

def _label(code: str, name: str) -> str:
    return f"{code} – {name}" if (name or "").strip() else code

def format_percent_display(value, metric_name: str) -> str:
    if pd.isna(value): return "—"
    return f"{value:.2f}%" if str(metric_name).strip().lower() == "52+ weeks" else f"{value:.1f}%"

def format_whole_round(x) -> str:
    if pd.isna(x): return "—"
    try:
        q = Decimal(str(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return f"{int(q):,}"
    except Exception:
        return "—"

def format_region_rank(rank_val, size_val) -> str:
    if pd.isna(rank_val) or pd.isna(size_val): return "—"
    return f"{int(rank_val)} out of {int(size_val)}"

def format_overall_rank(rank_val, nat_total) -> str:
    """Overall rank as 'X out of Y' (— if missing)."""
    if pd.isna(rank_val) or not nat_total:
        return "—"
    return f"{int(rank_val)} out of {int(nat_total)}"

def short_label(code: str, name: str, n_words: int = 3) -> str:
    words = (name or "").split()
    short = " ".join(words[:n_words]).strip()
    return f"{code} – {short}" if short else code

def weighted_percentage(sub: pd.DataFrame) -> float:
    num = pd.to_numeric(sub.get("numerator"), errors="coerce").fillna(0).sum()
    den = pd.to_numeric(sub.get("denominator"), errors="coerce").fillna(0).sum()
    if not den or pd.isna(den): return np.nan
    return float(np.clip(100.0 * num / den, 0.0, 100.0))

def ensure_percent_0_100(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    try:
        q = s.dropna().quantile(0.9)
        return s * 100.0 if q <= 1.5 else s
    except Exception:
        return s

# --- URL + memory persistence -----------------------------------------------
def _get_query_params():
    # Support both Streamlit query APIs across versions
    try:
        return dict(st.query_params)
    except Exception:
        return {
            k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in st.experimental_get_query_params().items()
        }

def _set_query_params(params: dict):
    """Merge new params into existing query string without dropping Streamlit's 'page'."""
    # Drop empties so we don't erase keys by writing blanks
    params = {k: v for k, v in params.items() if v not in (None, "", [])}

    try:
        # New API: dict-like; this preserves existing keys (incl. 'page')
        st.query_params.update(params)
    except Exception:
        # Old API: must merge explicitly, otherwise we'd overwrite everything
        current = st.experimental_get_query_params()
        # Flatten any singleton lists for consistency
        merged = {**current, **params}
        st.experimental_set_query_params(**merged)


# All the filters we care about across sessions/pages
FG_KEYS = ("freq", "domain", "metric", "region", "month", "quarter", "group")

def _save_filters_to_url():
    # Build only non-empty keys
    params = {}
    for k in ("freq", "domain", "metric", "region", "month", "quarter"):
        v = st.session_state.get(f"fg_{k}")
        if v not in (None, "", []):
            params[k] = v

    # Multiselect serialisation
    grp = st.session_state.get("fg_group", [])
    if grp:
        params["group"] = "|".join(grp)
    else:
        # Explicitly remove 'group' from the URL if selection is empty
        try:
            if "group" in st.query_params:
                del st.query_params["group"]
        except Exception:
            current = st.experimental_get_query_params()
            if "group" in current:
                current.pop("group", None)
                st.experimental_set_query_params(**current)

    # Merge the rest (preserves other keys incl. 'page')
    _set_query_params(params)

    # Keep a private memory copy (so empty stays empty)
    st.session_state["_fg_memory"] = {
        k: st.session_state.get(f"fg_{k}") for k in FG_KEYS
    }

def _save_fg_everywhere():
    # write to URL + keep a private memory copy
    _save_filters_to_url()
    st.session_state["_fg_memory"] = {
        k: st.session_state.get(f"fg_{k}") for k in FG_KEYS
    }

def _hydrate_fg_from_memory():
    """Seed fg_* keys from the last saved picks every rerun (non-destructive)."""
    mem = st.session_state.get("_fg_memory", {})
    for k in FG_KEYS:
        sk = f"fg_{k}"
        if sk not in st.session_state or st.session_state[sk] in (None, "", []):
            v = mem.get(k)
            if v not in (None, "", []):
                st.session_state[sk] = v


def _ensure_choice(key: str, options: list, fallback=None):
    # Make sure a select widget’s key has a valid value
    if not options:
        st.session_state[key] = None
        return None
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = (
            fallback if fallback in options else (options[-1] if options else None)
        )
    return st.session_state[key]

def _persist_multiselect(key: str, options: list, limit: int = 5):
    # Clamp multiselect values to current options and a max count
    cur = st.session_state.get(key, [])
    cur = [v for v in cur if v in options][:limit]
    st.session_state[key] = cur
    return cur
# ---------------------------------------------------------------------------

def build_kpi_table(df_slice: pd.DataFrame, codes: list[str], metric, nat_total: int | None) -> pd.DataFrame:
    rows = []
    order = df_slice.set_index("provider_code")["rank"] if "rank" in df_slice else None
    codes_sorted = sorted(codes, key=lambda c: order.get(c, 1e9)) if order is not None else codes

    for code in codes_sorted:
        r = df_slice.loc[df_slice["provider_code"] == code].head(1)
        if r.empty:
            continue
        r = r.iloc[0]

        rank_val       = r["rank"] if "rank" in r.index else np.nan
        rank_region    = r["rank_region"] if "rank_region" in r.index else np.nan
        region_size    = r["region_size"] if "region_size" in r.index else np.nan
        percent_val    = r["percent"] if "percent" in r.index else np.nan
        numerator_val  = r["numerator"] if "numerator" in r.index else np.nan
        denominator_val= r["denominator"] if "denominator" in r.index else np.nan

        rows.append({
            "Provider":     _label(r["provider_code"], r["provider_name"]),
            "Overall Rank": format_overall_rank(rank_val, nat_total),
            "Region Rank":  format_region_rank(rank_region, region_size),
            "% Value":      format_percent_display(percent_val, metric),
            "Numerator":    format_whole_round(numerator_val),
            "Denominator":  format_whole_round(denominator_val),
        })
    return pd.DataFrame(rows)

# ---- Filter state helpers (Foundation Group) ----
def _qp_get():
    """Safe get for query params across Streamlit versions."""
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _qp_set(**kwargs):
    """Safe set for query params across Streamlit versions."""
    try:
        # New API allows dict-like assignment
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

def _seed(key, value):
    """Only set a default if the key isn't already in session_state."""
    if key not in st.session_state:
        st.session_state[key] = value

def _first_valid(options, want, fallback_idx=0):
    """Return 'want' if it exists in options; else options[fallback_idx] or None."""
    if isinstance(options, (list, tuple)):
        return want if want in options else (options[fallback_idx] if options else None)
    return want

def render_metric_cards(tbl: pd.DataFrame, accent_color: str):
    """Render a responsive grid of metric cards for one provider."""
    if tbl.empty:
        st.caption("No rows under current filters.")
        return

    # ensure we have numeric % for the bar width
    pct = pd.to_numeric(tbl["percent"], errors="coerce").fillna(np.nan)
    pct = np.clip(pct, 0, 100)  # safety if data already in 0–100

    pieces = []
    for i, r in tbl.assign(_pct=pct).iterrows():
        title   = html.escape(str(r["metric"]))
        value   = format_percent_display(r["percent"], r["metric"])
        width   = "0%" if pd.isna(r["_pct"]) else f"{float(r['_pct']):.1f}%"
        rank    = "—" if pd.isna(r["rank"]) else f"{int(r['rank'])}"
        rr      = format_region_rank(r["rank_region"], r["region_size"])
        num     = format_whole_round(r["numerator"])
        den     = format_whole_round(r["denominator"])

        pieces.append(f"""
        <div class="metric-card">
          <div class="metric-top">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{accent_color}">{value}</div>
          </div>
          <div class="metric-meter" aria-hidden="true">
            <div class="metric-fill" style="width:{width}; background:{accent_color}"></div>
          </div>
          <div class="metric-footer">
            <span class="metric-chip">Overall Rank: {rank}</span>
            <span class="metric-chip">Region: {rr}</span>
            <span class="metric-chip">Num: {num}</span>
            <span class="metric-chip">Den: {den}</span>
          </div>
        </div>
        """)

    html_block = "<div class='metric-grid'>" + "".join(pieces) + "</div>"
    st.markdown(html_block, unsafe_allow_html=True)

def render_empty_state():
    # Context banner — same as Compare Providers (no .info-row wrapper)
    st.markdown(
        "<div id='context-banner'>Upload a <b>Monthly</b> or <b>Quarterly</b> CSV on the Homepage to use Foundation Group.</div>",
        unsafe_allow_html=True,
    )

    # Same padding rhythm as Compare Providers
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Two columns (no .info-row / .empty-grid wrappers)
    left, right = st.columns([0.50, 0.50], gap="medium")

    with left:
        st.markdown(
            """
            <div class="metrics-panel required-cols compact">
              <div class="metrics-panel-title">Get started</div>
              <ol>
                <li>Go to <b>Homepage</b> and upload a <b>Monthly</b> or <b>Quarterly</b> CSV.</li>
                <li>Return here and choose <b>Frequency</b>, pick the <b>Month/Quarter</b>, then select <b>Domain</b> and <b>Metric</b>.</li>
                <li>Pick up to <b>5 providers</b> in the sidebar (this selection only affects this page).</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="metrics-panel required-cols compact">
              <div class="metrics-panel-title">How this page works</div>
              <p>
                This page stacks KPIs for multiple providers in one table and highlights them in the
                <b>distribution</b> and <b>trend</b> charts. You can switch between <b>Monthly</b> and
                <b>Quarterly</b> data at the top of the sidebar. The <b>default provider</b> from other pages
                does not affect this page.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Space under the row (matches Compare Providers)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ---------------- Main code ----------------
monthly_bytes   = st.session_state.get("monthly_bytes")
quarterly_bytes = st.session_state.get("quarterly_bytes")
has_data = bool(monthly_bytes or quarterly_bytes)

# ---- Sidebar: "Data in use" like Compare Providers ----
with st.sidebar:
    st.header("📁 Data in use")
    if not has_data:
        st.warning("No files loaded. Upload on Homepage first.", icon="⚠️")
        if st.button("Go to Homepage", key="fg_goto_home"):
            st.switch_page("Homepage.py")

# ---- If no data, show the main-pane empty state and stop ----
if not has_data:
    render_empty_state()
    st.stop()

# Always (re)hydrate from URL if params are present
qp = _get_query_params()
for k in ("freq", "domain", "metric", "region", "month", "quarter"):
    v = qp.get(k)
    if v:
        st.session_state[f"fg_{k}"] = v

grp = qp.get("group")
# 👇 Ignore empty strings just in case
if isinstance(grp, str) and not grp.strip():
    grp = None

if grp:
    # Accept either legacy list (old API) or pipe-joined string (new API)
    if isinstance(grp, list):
        gvals = grp
    else:
        gvals = str(grp).split("|")
    st.session_state["fg_group"] = gvals


# Re-hydrate previously saved picks on every rerun (no "loaded" flag)
_hydrate_fg_from_memory()



# ---------------- Sidebar (only when data exists) -------------------------
if has_data:
    with st.sidebar:
        #st.markdown("### Foundation Group")
        _ensure_choice("fg_freq", ["Monthly", "Quarterly"], fallback="Monthly")
        freq = st.radio(
            "Frequency",
            ["Monthly", "Quarterly"],
            horizontal=True,
            key="fg_freq",
            on_change=_save_fg_everywhere
        )

# Read dataset
if freq == "Monthly":
    if monthly_bytes is None:
        render_empty_state(); st.stop()
    df = _read_csv_bytes(monthly_bytes)
    dt_us = pd.to_datetime(df["month"], dayfirst=False, errors="coerce")
    dt_uk = pd.to_datetime(df["month"], dayfirst=True,  errors="coerce")
    dt    = dt_us if dt_us.notna().sum() >= dt_uk.notna().sum() else dt_uk
    df["month_dt"]   = dt.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1) + pd.offsets.Day(1)
    df["month_disp"] = df["month_dt"].dt.strftime("%b-%y")
    period_col, period_label = "month_disp", "Month"
else:
    if quarterly_bytes is None:
        render_empty_state(); st.stop()
    df = _read_csv_bytes(quarterly_bytes)
    period_col, period_label = "quarter", "Quarter"

# Domain, Metric, Region
domains = sorted(x for x in df["domain"].dropna().unique())
with st.sidebar:
    _ensure_choice("fg_domain", domains, fallback=(domains[0] if domains else None))
    domain = st.selectbox(
        "Domain", domains,
        key="fg_domain",
        on_change=_save_fg_everywhere
    )

metrics = sorted(x for x in df.loc[df["domain"] == domain, "metric"].dropna().unique())
with st.sidebar:
    _ensure_choice("fg_metric", metrics, fallback=(metrics[0] if metrics else None))
    metric = st.selectbox(
        "Metric", metrics,
        key="fg_metric",
        on_change=_save_fg_everywhere
    )

regions = ["(All Regions)"] + sorted(x for x in df["region"].dropna().unique())
with st.sidebar:
    _ensure_choice("fg_region", regions, fallback="(All Regions)")
    region = st.selectbox(
        "Region", regions,
        key="fg_region",
        on_change=_save_fg_everywhere
    )

# Period
avail = df[(df["domain"] == domain) & (df["metric"] == metric)].copy()
if region != "(All Regions)":
    avail = avail[avail["region"] == region]

if freq == "Monthly":
    months = avail.sort_values("month_dt")["month_disp"].unique().tolist()
    with st.sidebar:
        _ensure_choice("fg_month", months, fallback=(months[-1] if months else None))
        month = st.selectbox(
            "Month", months,
            key="fg_month",
            on_change=_save_fg_everywhere
        )
    period = month
else:
    quarters = [q for q in pd.unique(avail["quarter"]) if pd.notna(q)]
    with st.sidebar:
        _ensure_choice("fg_quarter", quarters, fallback=(quarters[-1] if quarters else None))
        quarter = st.selectbox(
            "Quarter", quarters,
            key="fg_quarter",
            on_change=_save_fg_everywhere
        )
    period = quarter

# Providers (with sensible defaults)
providers_base = df if region == "(All Regions)" else df[df["region"] == region]
labels_df = _codes_and_names(providers_base)
labels = [_label(r.provider_code, r.provider_name) for _, r in labels_df.iterrows()]
code_by_label = {lab: labels_df.iloc[i]["provider_code"] for i, lab in enumerate(labels)}

# NEW: preselect 4 providers on first load (respect URL/session memory)
DEFAULT_CODES = ["RJC", "RLQ", "RLT", "RWP"]

# ⬇️ Only seed if the key does not exist at all (first load)
if "fg_group" not in st.session_state:
    code_to_label = {
        r.provider_code: _label(r.provider_code, r.provider_name)
        for _, r in labels_df.iterrows()
    }
    defaults = [code_to_label[c] for c in DEFAULT_CODES if c in code_to_label]
    if defaults:
        st.session_state["fg_group"] = defaults[:5]
        _save_fg_everywhere()

# keep your “clamp to options” behaviour
_persist_multiselect("fg_group", labels, limit=5)

with st.sidebar:
    st.caption("Select up to 5 providers (only for this page).")
    st.multiselect(
        "Foundation group",
        options=labels,
        key="fg_group",
        on_change=_save_fg_everywhere
    )


_save_fg_everywhere()

# Scope rows
if freq == "Monthly":
    scope = (df["domain"].eq(domain) & df["metric"].eq(metric) & df["month_disp"].eq(period))
else:
    scope = (df["domain"].eq(domain) & df["metric"].eq(metric) & df["quarter"].eq(period))
if region != "(All Regions)":
    scope &= df["region"].eq(region)
scoped = df.loc[scope].copy()

selected_codes = [code_by_label[l] for l in st.session_state.get("fg_group", [])]
selected_set = set(selected_codes)

# ---- Titles that include the selected metric ----
_metric_title = str(metric)
_unit_title = "month" if freq == "Monthly" else "quarter"
_trend_window = "last 24 months" if freq == "Monthly" else "last 8 quarters"
# --- heading helpers (match Compare Providers) ---
_metric_title = html.escape(str(metric))
_freq_label  = "Monthly" if freq == "Monthly" else "Quarterly"


# ---- Colors (moved up so KPI table can reuse them) ----
BAR_GREY = "#DDE3EA"
PALETTE = ["#327AD1", "#AE2573", "#0C988F", "#F47738", "#85994B"]
REG_LINE = "#6B7280"
NAT_LINE = "#0C988F"

# Keep the same order mapping used by the charts
colour_by_code = {code: PALETTE[i % len(PALETTE)] for i, code in enumerate(selected_codes)}

has_rows = not scoped.empty
if not has_rows:
    st.markdown(
        """
        <div class='full-bleed'>
        <div class='info-banner'>
            No rows for this Domain/Metric/Period/Region. Try another period or region.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Header ----------------
st.markdown(
    f"""
    <div class="app-header" role="banner" aria-label="Header">
      <div class="app-logo" aria-hidden="true">{logo_svg}</div>
      <h1 id="page-title">NOF Foundation Group</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

ctx_bits = [f"{html.escape(domain)}", f"{html.escape(metric)}", f"{html.escape(period)}"]
if region != "(All Regions)":
    ctx_bits.append(f"{html.escape(region)}")
st.markdown(
    f"<div id='context-banner'>Viewing <b>{' → '.join(ctx_bits)}</b>.</div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ========================= KPI Table ======================================
if has_rows:
    st.markdown(f"##### KPIs — {_metric_title} (selected providers)", unsafe_allow_html=True)
    # National scope for Y (ignore Region; same Domain + Metric + Period)
    nat_scope = df[
        (df["domain"] == domain) &
        (df["metric"] == metric) &
        (df[period_col] == period)
    ]
    nat_total = nat_scope["provider_code"].nunique()

    kpi_df = build_kpi_table(scoped, selected_codes, metric, nat_total) if selected_codes else pd.DataFrame(
        columns=["Provider","Overall Rank","Region Rank","% Value","Numerator","Denominator"]
    )

    if not kpi_df.empty:
        col_order = ["Provider","Overall Rank","Region Rank","% Value","Numerator","Denominator"]
        kpi_df = kpi_df[col_order]

        # per-row colors for the Provider column
        provider_codes = [str(v).split("–")[0].strip() for v in kpi_df["Provider"]]
        provider_text_colors = [colour_by_code.get(c, "#111827") for c in provider_codes]

        # BOLD provider names
        provider_labels_bold = [f"<b>{v}</b>" for v in kpi_df["Provider"]]

        fig_kpi = go.Figure(data=[go.Table(
            columnwidth=[300, 100, 100, 100, 100, 100],
            header=dict(
                values=col_order, fill_color="#F3F4F6", align="left",
                font=dict(size=12, color="#111827")
            ),
            cells=dict(
                values=[
                    provider_labels_bold,                 # ← bold text
                    kpi_df["Overall Rank"],
                    kpi_df["Region Rank"],
                    kpi_df["% Value"],
                    kpi_df["Numerator"],
                    kpi_df["Denominator"],
                ],
                align="left",
                font=dict(color=[
                    provider_text_colors,  # colored provider text
                    "#111827", "#111827", "#111827", "#111827", "#111827"
                ]),
                height=28
            )
        )])

        # Collapse the blank space below the table by setting an explicit height
        _header_h = 36
        _row_h = 28
        fig_kpi.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=_header_h + _row_h * len(kpi_df) + 4
        )

        st.plotly_chart(fig_kpi, use_container_width=True, config={"displaylogo": False})

st.markdown("<div class='no-gap'><hr class='section-sep'></div>", unsafe_allow_html=True)


# =========================== Distribution Chart ==================================

bars = scoped.copy()
bars["is_selected"] = bars["provider_code"].isin(selected_set)
colour_by_code = {code: PALETTE[i % len(PALETTE)] for i, code in enumerate(selected_codes)}
bars["colour"] = bars["provider_code"].map(lambda c: colour_by_code.get(c, BAR_GREY))

bars["PercentLabel"]    = bars.apply(lambda r: format_percent_display(r["percent"], r["metric"]), axis=1)
bars["NumLabel"]        = bars["numerator"].map(format_whole_round)
bars["DenLabel"]        = bars["denominator"].map(format_whole_round)
bars["RegionRankLabel"] = bars.apply(lambda r: format_region_rank(r["rank_region"], r["region_size"]), axis=1)

fig_dist = px.bar(
    bars.sort_values("rank", na_position="last"),
    x="provider_code", y="percent",
    height=560,
)
fig_dist.update_traces(
    marker_color=bars.sort_values("rank", na_position="last")["colour"],
    customdata=bars.sort_values("rank", na_position="last")[[
        "provider_code","provider_name","region",
        "NumLabel","DenLabel","PercentLabel","rank","RegionRankLabel"
    ]].values,
    hovertemplate=(
        "<b>%{customdata[0]}</b> – %{customdata[1]}<br>"
        "Region: %{customdata[2]}<br>"
        "Numerator: %{customdata[3]}<br>"
        "Denominator: %{customdata[4]}<br>"
        "% Value: %{customdata[5]}<br>"
        "Rank: %{customdata[6]}<br>"
        "Region rank: %{customdata[7]}<extra></extra>"
    ),
)
fig_dist.update_layout(
    xaxis_title="Providers",
    yaxis_title=None,
    margin=dict(l=10, r=10, t=10, b=10),
    height=400
)
fig_dist.update_xaxes(showticklabels=False)
fig_dist.update_yaxes(showticklabels=False)

# AFTER (uniform H5 + metric in title, exactly like Compare Providers)
_period_word = "month" if freq == "Monthly" else "quarter"
st.markdown(f"##### {html.escape(str(metric))} — Distribution — selected {_period_word}", unsafe_allow_html=True)
st.plotly_chart(fig_dist, use_container_width=True, config={"displaylogo": False})

# Separator between charts and the "Metrics within domain" section
st.markdown("<div class='no-gap'><hr class='section-sep'></div>", unsafe_allow_html=True)

# =========================== Trend Chart ==================================

# === Trend (left) + Metrics within domain (right) ===

left, right = st.columns([0.58, 0.42], gap="small")  # match Compare Providers

with left:
    # ---- Title for the trend chart (same wording as other pages)
    st.markdown(
        f"##### Trend — {html.escape(str(metric))} "
        f"({'last 24 months' if freq=='Monthly' else 'last 8 quarters'})",
        unsafe_allow_html=True
    )

    import plotly.graph_objects as go
    fig_line = go.Figure()
    HOVER_TMPL = "<b>%{fullData.name}</b><br>%{customdata[0]}<extra></extra>"

    if freq == "Monthly":
        dm = df[(df["domain"] == domain) & (df["metric"] == metric)].copy()
        dm = dm.sort_values("month_dt").drop_duplicates(subset=["month_dt", "provider_code"], keep="last")
        latest = dm["month_dt"].max()
        months24 = dm[["month_dt"]].drop_duplicates().sort_values("month_dt")["month_dt"]
        months24 = months24[months24 <= latest].tail(24).tolist()

        # provider series
        for i, code in enumerate(selected_codes):
            s = dm[dm["provider_code"] == code].sort_values("month_dt")
            if s.empty:
                continue
            s["PercentLabel"] = s.apply(lambda r: format_percent_display(r["percent"], r["metric"]), axis=1)
            fig_line.add_scatter(
                x=s["month_dt"], y=s["percent"],
                mode="lines+markers",
                name=short_label(code, s["provider_name"].iloc[0] if "provider_name" in s.columns else "", 3),
                line=dict(width=3, color=colour_by_code.get(code, PALETTE[i % len(PALETTE)])),
                marker=dict(size=6, line=dict(width=1, color="#FFFFFF")),
                customdata=s[["PercentLabel"]].values,
                hovertemplate=HOVER_TMPL,
            )

        # region / national weighted
        def _add_weighted(frame: pd.DataFrame, name: str, dash: str):
            xs, ys = [], []
            for mdt in months24:
                xs.append(mdt)
                ys.append(weighted_percentage(frame[frame["month_dt"] == mdt]))
            fig_line.add_scatter(
                x=xs, y=ys, name=name, mode="lines",
                line=dict(width=2, dash=dash, color=REG_LINE if dash == "dash" else NAT_LINE),
                customdata=[[format_percent_display(y, metric)] for y in ys],
                hovertemplate=HOVER_TMPL,
            )
        if region != "(All Regions)":
            _add_weighted(dm[dm["region"] == region], f"{region} (weighted)", "dash")
        _add_weighted(dm, "National (weighted)", "dot")

        fig_line.update_xaxes(title=None, tickformat="%b-%y")

    else:
        dq = df[(df["domain"] == domain) & (df["metric"] == metric)].copy()
        order = pd.Categorical(dq["quarter"], categories=pd.unique(dq["quarter"]), ordered=True)
        dq["quarter"] = order
        last8 = list(order.categories)[-8:]
        dq8 = dq[dq["quarter"].isin(last8)]

        for i, code in enumerate(selected_codes):
            s = dq8[dq8["provider_code"] == code].copy()
            if s.empty:
                continue
            s["PercentLabel"] = s.apply(lambda r: format_percent_display(r["percent"], r["metric"]), axis=1)
            fig_line.add_scatter(
                x=s["quarter"].astype(str), y=s["percent"],
                mode="lines+markers",
                name=short_label(code, s["provider_name"].iloc[0] if "provider_name" in s.columns else "", 3),
                line=dict(width=3, color=colour_by_code.get(code, PALETTE[i % len(PALETTE)])),
                marker=dict(size=6, line=dict(width=1, color="#FFFFFF")),
                customdata=s[["PercentLabel"]].values,
                hovertemplate=HOVER_TMPL,
            )

        def _add_weighted_q(frame: pd.DataFrame, name: str, dash: str):
            xs, ys = [], []
            for q in last8:
                xs.append(str(q))
                ys.append(weighted_percentage(frame[frame["quarter"] == q]))
            fig_line.add_scatter(
                x=xs, y=ys, name=name, mode="lines",
                line=dict(width=2, dash=dash, color=REG_LINE if dash == "dash" else NAT_LINE),
                customdata=[[format_percent_display(y, metric)] for y in ys],
                hovertemplate=HOVER_TMPL,
            )
        if region != "(All Regions)":
            _add_weighted_q(dq8[dq8["region"] == region], f"{region} (weighted)", "dash")
        _add_weighted_q(dq8, "National (weighted)", "dot")

        fig_line.update_xaxes(title=None)  # quarter labels

    fig_line.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=6, b=0),   # small, to align with right panel
        legend=dict(orientation="h", y=-0.15, x=-0.05),
        hovermode="x unified",
        showlegend=False,
    )
    fig_line.update_yaxes(ticksuffix="%", title=None)
    st.plotly_chart(fig_line, use_container_width=True, config={"displaylogo": False})

with right:
    # ---- Title + provider tabs
    st.markdown(
        f"##### Metrics within domain — {html.escape(str(domain))} ({_freq_label})",
        unsafe_allow_html=True
    )

    if selected_codes:
        name_by_code = {r.provider_code: r.provider_name for _, r in labels_df.iterrows()}
        tabs = st.tabs([short_label(c, name_by_code.get(c, ""), 2) for c in selected_codes])

        for i, code in enumerate(selected_codes):
            with tabs[i]:
                tbl = df[(df[period_col]==period) & (df["domain"]==domain)]
                if region != "(All Regions)":
                    tbl = tbl[tbl["region"]==region]
                tbl = tbl[tbl["provider_code"]==code][
                    ["metric","percent","rank","rank_region","region_size","numerator","denominator"]
                ].copy().sort_values("metric")

                if tbl.empty:
                    st.caption("No rows under current filters.")
                else:
                    # Build vertical progress-cards (matches Compare Providers)
                    cards = []
                    for _, r in tbl.iterrows():
                        # % width for the meter (0–100), safe if CSV is 0–1 or 0–100
                        pct = pd.to_numeric(r["percent"], errors="coerce")
                        pct_val = 0.0 if pd.isna(pct) else float(np.clip(pct, 0.0, 100.0))

                        title     = html.escape(str(r["metric"]))
                        pct_label = format_percent_display(r["percent"], r["metric"])

                        # National “X out of Y” for this metric (ignore Region)
                        nat_total_m = df[
                            (df[period_col] == period) &
                            (df["domain"] == domain) &
                            (df["metric"] == r["metric"])
                        ]["provider_code"].nunique()
                        rank_txt  = "—" if pd.isna(r["rank"]) else f"{int(r['rank'])}"
                        nat_label = "—" if (rank_txt == "—" or not nat_total_m) else f"{rank_txt} out of {int(nat_total_m)}"

                        # Regional “X out of Y”
                        if pd.isna(r["rank_region"]) and pd.isna(r["region_size"]):
                            reg_txt = "—"
                        else:
                            rr = "—" if pd.isna(r["rank_region"]) else f"{int(r['rank_region'])}"
                            reg_txt = rr if pd.isna(r["region_size"]) else f"{rr} out of {int(r['region_size'])}"

                        card_html = dedent(f"""
                        <div class="progress-card">
                        <div class="progress-head">
                            <div class="progress-name">{title}</div>
                            <div class="progress-percent">{pct_label}</div>
                        </div>
                        <div class="progress-track"><div class="progress-fill" style="width:{pct_val:.2f}%;"></div></div>
                        <div class="progress-caption">
                            <span>{nat_label}</span>
                            <span class="muted">Regional Rank: {reg_txt}</span>
                        </div>
                        </div>
                        """).strip()

                        cards.append(card_html)

                    panel_html = dedent("""
                    <div class="metrics-panel cards-plain">
                    <div class="progress-list">{cards}</div>
                    </div>
                    """).format(cards="".join(cards)).strip()

                    st.markdown(panel_html, unsafe_allow_html=True)

    else:
        st.caption("Select providers in the sidebar to view per-provider metrics.")

