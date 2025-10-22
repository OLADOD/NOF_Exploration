Cause
You need a fast way to run or deploy the dashboard and know where to tweak it.

Intention
Give you one clear set of steps that works on Windows/Mac/Linux and Streamlit Cloud.

Solution

A) Setup locally in VS Code

Open a terminal in your project folder.

Create a virtual env (choose the one for your OS):

Windows (PowerShell):

python -m venv .venv
.\.venv\Scripts\Activate.ps1


macOS/Linux (bash/zsh):

python -m venv .venv
source .venv/bin/activate


Install packages:

pip install -r requirements.txt


Run the app:

streamlit run app.py


In the browser, upload your monthly CSV (must include these columns exactly, case-sensitive):
Quarter, Domain, Metric, Region, Provider Code, Provider Name, Numerator, Denominator, % Value, Rank, Months Covered, Covered Months
✔ % Value should be a percent string like 98.3%.
✔ If your metric is 52+ Weeks, it will display at 2 decimal places; others at 1.
✔ Missing values are hidden in charts and shown as --- in cards/table.

Use the sidebar filters (top-down): Quarter → Domain → Metric → Region → Provider.

Region set = chart shows only that region; Region not set = all regions.

Provider selection does not change the chart, only highlights the bar and shows KPI cards.

If no Provider is chosen, the app tries to default to RWP when available.

See data table under “See filtered data…” and download CSV/Excel.

B) Deploy to Streamlit Community Cloud (free)

Put these files in a GitHub repo:

app.py

requirements.txt

.streamlit/config.toml (folder + file)

README.md (optional)

Go to share.streamlit.io → New app → connect your repo → choose branch and app.py.

Deploy. When it opens, use the Upload button to provide each month’s CSV.

C) Common tweaks (open app.py and edit)

Default provider code: change DEFAULT_PROVIDER_CODE = "RWP".

Highlight color: change HIGHLIGHT_HEX = "#FAE100".

Bar color for others: change BAR_NEUTRAL_HEX.

Change columns shown in the table: edit the COLS_USED list.

Round bars height: adjust cornerRadiusTopLeft/TopRight in build_chart().

Percent formatting rule: update format_percent_display().