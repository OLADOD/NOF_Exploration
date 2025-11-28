<#
.SYNOPSIS
    Sets up the Streamlit application as a Windows Service using NSSM.

.DESCRIPTION
    This script installs Chocolatey (if not present), installs NSSM via Chocolatey,
    and configures a Windows Service to run the Streamlit application.

.NOTES
    Run this script as Administrator.
#>

# Ensure the script is run as Administrator
if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "You do not have Administrator rights to run this script!`nPlease re-run this script as an Administrator!"
    Break
}

# --- Configuration ---
$ServiceName = "NOF_Dashboards"
$DisplayName = "NOF Dashboards Streamlit App"
$Description = "Runs the NOF Dashboards Streamlit application."
$AppScript = "Homepage.py"
$AppPort = 8501

# Get the script's directory (assumed to be repo root)
$RepoRoot = $PSScriptRoot

# Path to Python executable in the virtual environment
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# check if venv exists
if (-not (Test-Path $PythonPath)) {
    Write-Warning "Virtual environment python not found at: $PythonPath"
    $PythonPath = Read-Host "Please enter the full path to your Python executable"
    if (-not (Test-Path $PythonPath)) {
        Write-Error "Python executable not found. Please create a .venv or provide a valid path."
        Exit 1
    }
}

Write-Host "Using Python: $PythonPath"
Write-Host "Repo Root: $RepoRoot"

# --- Install Chocolatey (if not installed) ---
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey not found. Installing..."
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
} else {
    Write-Host "Chocolatey is already installed."
}

# --- Install NSSM ---
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Host "NSSM not found. Installing via Chocolatey..."
    choco install nssm -y
    # Refresh env vars so nssm is available in current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Host "NSSM is already installed."
}

# Check if nssm is now available
if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
    Write-Error "NSSM installation failed or not found in PATH. Please restart the shell or install manually."
    Exit 1
}

# --- Configure Service ---

# Check if service already exists
$ServiceStatus = Get-Service $ServiceName -ErrorAction SilentlyContinue
if ($ServiceStatus) {
    Write-Host "Service '$ServiceName' already exists. Removing it..."
    nssm stop $ServiceName
    nssm remove $ServiceName confirm
}

Write-Host "Installing Service '$ServiceName'..."

# Arguments for Streamlit
# Note: absolute paths for script
$ScriptPath = Join-Path $RepoRoot $AppScript
$Arguments = "-m streamlit run `"$ScriptPath`" --server.port $AppPort --server.headless true"

# Install service
nssm install $ServiceName "$PythonPath" $Arguments

# Set additional parameters
nssm set $ServiceName DisplayName "$DisplayName"
nssm set $ServiceName Description "$Description"
nssm set $ServiceName AppDirectory "$RepoRoot"
nssm set $ServiceName Start SERVICE_AUTO_START

# Logging (Redirect stdout/stderr)
$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$StdoutLog = Join-Path $LogDir "service_stdout.log"
$StderrLog = Join-Path $LogDir "service_stderr.log"

nssm set $ServiceName AppStdout "$StdoutLog"
nssm set $ServiceName AppStderr "$StderrLog"
# Enable log rotation (optional but recommended)
nssm set $ServiceName AppRotateFiles 1
nssm set $ServiceName AppRotateOnline 1
nssm set $ServiceName AppRotateSeconds 86400
nssm set $ServiceName AppRotateBytes 5242880

# --- Start Service ---
Write-Host "Starting Service '$ServiceName'..."
nssm start $ServiceName

Write-Host "Service setup complete!"
Write-Host "You can access the app at http://localhost:$AppPort"

Read-Host -Prompt "Press Enter to exit"