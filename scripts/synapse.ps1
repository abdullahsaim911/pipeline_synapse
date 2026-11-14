param(
    [ValidateSet("start", "status", "stop", "setup")]
    [string]$Command = "start",
    [string]$Python311 = "",
    [switch]$RequireSignIn,
    [switch]$NoLogWindow
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$PidFile = Join-Path $RuntimeRoot "backend.pid"
$SupervisorPidFile = Join-Path $RuntimeRoot "supervisor.pid"
$LogViewerPidFile = Join-Path $RuntimeRoot "log-viewer.pid"
$StopFile = Join-Path $RuntimeRoot "stop.requested"
$ApiUrl = "http://127.0.0.1:8000"

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE. No runtime was marked ready."
    }
}

function Ensure-CudaTorchWheel([string]$Wheelhouse) {
    $wheelName = "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
    $wheelPath = Join-Path $Wheelhouse $wheelName
    $expectedSha256 = "6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb"
    if (Test-Path -LiteralPath $wheelPath) {
        $actual = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $expectedSha256) { return }
        Remove-Item -LiteralPath $wheelPath -Force
    }

    $partialPath = "$wheelPath.partial"
    $url = "https://download-r2.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl"
    Write-Host "Downloading the pinned CUDA runtime with durable resume support..."
    & curl.exe --fail --location --retry 100 --retry-all-errors --retry-delay 2 `
      --connect-timeout 30 --speed-limit 1024 --speed-time 60 `
      --continue-at - --output $partialPath $url
    Assert-NativeSuccess "Downloading the CUDA runtime"
    $actual = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expectedSha256) {
        throw "The downloaded CUDA runtime failed checksum verification."
    }
    Move-Item -LiteralPath $partialPath -Destination $wheelPath -Force
}

function Test-Endpoint([string]$Path) {
    try {
        return Invoke-RestMethod -Uri "$ApiUrl$Path" -TimeoutSec 2
    } catch {
        # /ready returns useful component states with HTTP 503 while local
        # workers load. Keep those states visible in the launcher's status log.
        if ($Path -eq "/ready" -and $_.ErrorDetails.Message) {
            try {
                $readiness = $_.ErrorDetails.Message | ConvertFrom-Json
                if ($null -ne $readiness.components) { return $readiness }
            } catch {}
        }
        return $null
    }
}

function Start-LogViewer {
    if ($NoLogWindow) { return }
    $viewer = $null
    if (Test-Path -LiteralPath $LogViewerPidFile) {
        $viewerPid = 0
        [int]::TryParse((Get-Content -LiteralPath $LogViewerPidFile -Raw), [ref]$viewerPid) | Out-Null
        if ($viewerPid -gt 0) {
            $viewer = Get-Process -Id $viewerPid -ErrorAction SilentlyContinue
        }
    }
    if ($null -ne $viewer) { return }
    Remove-Item -LiteralPath $LogViewerPidFile -Force -ErrorAction SilentlyContinue

    $viewerParentPid = $PID
    foreach ($candidateFile in @($SupervisorPidFile, $PidFile)) {
        if (-not (Test-Path -LiteralPath $candidateFile)) { continue }
        $candidatePid = 0
        [int]::TryParse((Get-Content -LiteralPath $candidateFile -Raw), [ref]$candidatePid) | Out-Null
        if ($candidatePid -gt 0 -and (Get-Process -Id $candidatePid -ErrorAction SilentlyContinue)) {
            $viewerParentPid = $candidatePid
            break
        }
    }

    $watcherScript = Join-Path $PSScriptRoot "watch-backend-logs.ps1"
    $viewer = Start-Process -FilePath "powershell.exe" -WindowStyle Normal -PassThru -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $watcherScript,
        "-LogPath", (Join-Path $RuntimeRoot "backend.err.log"),
        "-ParentPid", $viewerParentPid
    )
    Set-Content -LiteralPath $LogViewerPidFile -Value $viewer.Id
}

function Stop-LogViewer {
    if (Test-Path -LiteralPath $LogViewerPidFile) {
        $viewerPid = 0
        [int]::TryParse((Get-Content -LiteralPath $LogViewerPidFile -Raw), [ref]$viewerPid) | Out-Null
        if ($viewerPid -gt 0) {
            $viewer = Get-Process -Id $viewerPid -ErrorAction SilentlyContinue
            if ($null -ne $viewer) { Stop-Process -Id $viewerPid -ErrorAction SilentlyContinue }
        }
        Remove-Item -LiteralPath $LogViewerPidFile -Force -ErrorAction SilentlyContinue
    }
}

function Get-RuntimePython([string]$Name) {
    $isolated = Join-Path $RuntimeRoot "$Name\Scripts\python.exe"
    $readyMarker = Join-Path $RuntimeRoot "$Name\.ready"
    if ((Test-Path -LiteralPath $isolated) -and (Test-Path -LiteralPath $readyMarker)) {
        return $isolated
    }
    $legacy = Join-Path $RepoRoot "synapse-env\Scripts\python.exe"
    if (Test-Path -LiteralPath $legacy) { return $legacy }
    throw "No Synapse Python runtime found. Run '.\scripts\synapse.ps1 setup -Python311 <python.exe>'."
}

function Show-Status {
    $live = Test-Endpoint "/health"
    if ($null -eq $live) {
        Write-Host "Synapse backend: stopped"
        return
    }
    $ready = Test-Endpoint "/ready"
    if ($null -ne $ready) {
        Write-Host "Synapse backend: $($ready.status)"
        foreach ($entry in $ready.components.PSObject.Properties) {
            Write-Host "  $($entry.Name): $($entry.Value)"
        }
    } else {
        Write-Host "Synapse backend: alive (workers are still warming)"
    }
}

function Get-ReadinessSignature($Readiness) {
    if ($null -eq $Readiness) { return "unavailable" }
    return (($Readiness.components.PSObject.Properties | ForEach-Object {
        "$($_.Name)=$($_.Value)"
    }) -join ";")
}

function Show-ModelTransition($Readiness) {
    Write-Host "Model readiness changed:"
    foreach ($entry in $Readiness.components.PSObject.Properties) {
        Write-Host "  $($entry.Name): $($entry.Value)"
    }
    $components = $Readiness.components
    $modalConfiguredReady = $components.modal -in @("ready", "disabled")
    $mistralConfiguredReady = $components.mistral -in @("ready", "disabled", $null)
    if (
        $components.voice -eq "ready" -and
        $components.transcription -eq "ready" -and
        $modalConfiguredReady -and $mistralConfiguredReady
    ) {
        Write-Host "All configured startup models are ready. The pipeline can be started."
    } elseif ($components.modal -eq "unavailable") {
        Write-Warning "Modal VLM is unavailable. The API remains alive and VLM work will wait and retry safely."
    } elseif ($components.mistral -eq "unavailable") {
        Write-Warning "Remote Mistral is unavailable. The API remains alive; explanation requests report provider failures without changing models."
    }
}

if ($Command -eq "status") {
    Show-Status
    exit 0
}

if ($Command -eq "stop") {
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    Set-Content -LiteralPath $StopFile -Value "stop"
    if (-not (Test-Path -LiteralPath $PidFile)) {
        Write-Host "No launcher-owned backend is recorded."
    } else {
        $backendPid = [int](Get-Content -LiteralPath $PidFile -Raw)
        $process = Get-Process -Id $backendPid -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Stop-Process -Id $backendPid
            [void]$process.WaitForExit(5000)
            if (-not $process.HasExited) { Stop-Process -Id $backendPid -Force }
        }
    }
    if (Test-Path -LiteralPath $SupervisorPidFile) {
        $supervisorPid = [int](Get-Content -LiteralPath $SupervisorPidFile -Raw)
        if ($supervisorPid -ne $PID) {
            $supervisor = Get-Process -Id $supervisorPid -ErrorAction SilentlyContinue
            if ($null -ne $supervisor) {
                [void]$supervisor.WaitForExit(7000)
                if (-not $supervisor.HasExited) { Stop-Process -Id $supervisorPid }
            }
        }
    }
    Stop-LogViewer
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $SupervisorPidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
    Write-Host "Synapse backend stopped."
    exit 0
}

if ($Command -eq "setup") {
    if ([string]::IsNullOrWhiteSpace($Python311)) {
        $localPython311 = Join-Path $RuntimeRoot "python311\python.exe"
        if (Test-Path -LiteralPath $localPython311) {
            $Python311 = $localPython311
        } else {
            throw "Install Python 3.11 inside .runtime or pass its path with -Python311."
        }
    }
    $version = & $Python311 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    Assert-NativeSuccess "Python version check"
    if ($version -ne "3.11") { throw "The runtime installer requires Python 3.11, received $version." }
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    # Large CUDA wheels can exceed the free space on the Windows system drive.
    # Keep setup downloads, build files, and pip's cache inside the repository
    # runtime directory where the installed environments already live.
    $setupTemp = Join-Path $RuntimeRoot "tmp"
    $setupCache = Join-Path $RuntimeRoot "pip-cache"
    $wheelhouse = Join-Path $RuntimeRoot "wheels"
    New-Item -ItemType Directory -Path $setupTemp -Force | Out-Null
    New-Item -ItemType Directory -Path $setupCache -Force | Out-Null
    New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null
    $env:TEMP = $setupTemp
    $env:TMP = $setupTemp
    $env:PIP_CACHE_DIR = $setupCache
    Ensure-CudaTorchWheel $wheelhouse
    foreach ($name in @("api", "voice", "pipeline")) {
        $target = Join-Path $RuntimeRoot $name
        $readyMarker = Join-Path $target ".ready"
        Remove-Item -LiteralPath $readyMarker -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path -LiteralPath "$target\Scripts\python.exe")) {
            & $Python311 -m venv $target
            Assert-NativeSuccess "Creating the $name runtime"
        }
        & "$target\Scripts\python.exe" -m pip install --upgrade pip
        Assert-NativeSuccess "Updating pip in the $name runtime"
        if ($name -in @("api", "pipeline")) {
            $cudaWheel = Join-Path $wheelhouse "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
            & "$target\Scripts\python.exe" -m pip install --no-index --no-deps $cudaWheel
            Assert-NativeSuccess "Installing the pinned CUDA runtime in the $name runtime"
        }
        $installArgs = @(
            "-m", "pip", "install", "--resume-retries", "30",
            "-r", (Join-Path $RepoRoot "requirements\runtime-$name.lock")
        )
        if ($name -in @("api", "pipeline")) {
            $installArgs += @("--find-links", $wheelhouse)
        }
        & "$target\Scripts\python.exe" @installArgs
        Assert-NativeSuccess "Installing the $name runtime"
        & "$target\Scripts\python.exe" -m pip check
        Assert-NativeSuccess "Checking the $name runtime"
    }
    Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
    Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
    & (Join-Path $RuntimeRoot "api\Scripts\python.exe") `
      (Join-Path $RepoRoot "scripts\cache_models.py") api
    Assert-NativeSuccess "Caching SpeechT5 fallback models"
    & (Join-Path $RuntimeRoot "voice\Scripts\python.exe") `
      (Join-Path $RepoRoot "scripts\cache_models.py") voice
    Assert-NativeSuccess "Caching voice models"
    & (Join-Path $RuntimeRoot "pipeline\Scripts\python.exe") `
      (Join-Path $RepoRoot "scripts\cache_models.py") pipeline
    Assert-NativeSuccess "Caching pipeline models"
    foreach ($name in @("api", "voice", "pipeline")) {
        Set-Content -LiteralPath (Join-Path $RuntimeRoot "$name\.ready") -Value "ready"
    }
    Write-Host "Synapse runtimes are installed. Use '.\scripts\synapse.ps1 start'."
    exit 0
}

if ($null -ne (Test-Endpoint "/health")) {
    if ($RequireSignIn) {
        throw "A backend is already running; its account policy was not changed. Stop it with '.\scripts\synapse.ps1 stop', then start again with -RequireSignIn."
    }
    Write-Host "A Synapse backend is already running; no duplicate was started."
    Start-LogViewer
    Show-Status
    exit 0
}

$apiPython = Get-RuntimePython "api"
# This launcher defaults to the developer workflow. Built/packaged use starts
# the same local API with -RequireSignIn; direct API starts default to protected.
if ($RequireSignIn) { $env:SYNAPSE_LOCAL_PROFILE_COMPAT = "0" }
elseif ([string]::IsNullOrWhiteSpace($env:SYNAPSE_LOCAL_PROFILE_COMPAT)) {
    $env:SYNAPSE_LOCAL_PROFILE_COMPAT = "1"
}
$env:SYNAPSE_VOICE_PYTHON = Get-RuntimePython "voice"
$env:SYNAPSE_PIPELINE_PYTHON = Get-RuntimePython "pipeline"
# Let the backend's .env loader resolve GPU layers and context. Python owns
# their CPU/context defaults; explicit inherited environment values still win.
if ([string]::IsNullOrWhiteSpace($env:SYNAPSE_OLLAMA_KEEP_ALIVE)) { $env:SYNAPSE_OLLAMA_KEEP_ALIVE = "5m" }
if ([string]::IsNullOrWhiteSpace($env:SYNAPSE_TRANSCRIPTION_DEVICE)) { $env:SYNAPSE_TRANSCRIPTION_DEVICE = "cpu" }
if ([string]::IsNullOrWhiteSpace($env:SYNAPSE_VOICE_EMBED_BATCH_SIZE)) { $env:SYNAPSE_VOICE_EMBED_BATCH_SIZE = "16" }
if ([string]::IsNullOrWhiteSpace($env:OMP_NUM_THREADS)) { $env:OMP_NUM_THREADS = "1" }
if ([string]::IsNullOrWhiteSpace($env:OPENBLAS_NUM_THREADS)) { $env:OPENBLAS_NUM_THREADS = "1" }
if ([string]::IsNullOrWhiteSpace($env:MKL_NUM_THREADS)) { $env:MKL_NUM_THREADS = "1" }
if ([string]::IsNullOrWhiteSpace($env:SYNAPSE_LOCAL_MODELS_ONLY)) { $env:SYNAPSE_LOCAL_MODELS_ONLY = "1" }
if ($env:SYNAPSE_LOCAL_MODELS_ONLY -eq "1") {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}
$env:PYTHONUNBUFFERED = "1"

New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $SupervisorPidFile -Value $PID
$stdoutLog = Join-Path $RuntimeRoot "backend.out.log"
$stderrLog = Join-Path $RuntimeRoot "backend.err.log"
$restartCount = 0
$maximumRestarts = 3
try {
    while ($restartCount -le $maximumRestarts) {
        Start-LogViewer
        $backend = Start-Process -FilePath $apiPython -ArgumentList @(
            "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"
        ) -WorkingDirectory $RepoRoot -NoNewWindow -PassThru `
          -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
        Set-Content -LiteralPath $PidFile -Value $backend.Id

        $deadline = (Get-Date).AddSeconds(30)
        while (-not $backend.HasExited -and (Get-Date) -lt $deadline) {
            if ($null -ne (Test-Endpoint "/health")) { break }
            Start-Sleep -Milliseconds 250
        }
        if (-not $backend.HasExited) {
            Write-Host "Backend is alive. Voice, transcription, VLM and remote Mistral warm independently when configured."
            Show-Status
        }
        $lastReadinessSignature = ""
        while (-not $backend.WaitForExit(2000)) {
            $readiness = Test-Endpoint "/ready"
            $readinessSignature = Get-ReadinessSignature $readiness
            if ($readinessSignature -ne $lastReadinessSignature) {
                $lastReadinessSignature = $readinessSignature
                if ($null -ne $readiness) {
                    Show-ModelTransition $readiness
                }
            }
        }
        if (Test-Path -LiteralPath $StopFile) { break }
        if ($backend.ExitCode -eq 0) { break }
        $restartCount += 1
        if ($restartCount -gt $maximumRestarts) { break }
        Write-Warning "Backend exited unexpectedly. Restarting ($restartCount/$maximumRestarts)..."
        Start-Sleep -Seconds ([Math]::Min(5, $restartCount * 2))
    }
} finally {
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $SupervisorPidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
}

if ($restartCount -gt $maximumRestarts) {
    throw "Backend stopped repeatedly. Inspect the sanitized backend log before retrying."
}
