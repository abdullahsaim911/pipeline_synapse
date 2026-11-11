param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [int]$ParentPid
)

$ErrorActionPreference = "Continue"
Write-Host "Synapse detailed pipeline logs"
Write-Host "Watching: $LogPath"
Write-Host "Close this window with Ctrl+C; the backend will keep running."
Write-Host ""

$position = 0L
while ($true) {
    $parent = Get-Process -Id $ParentPid -ErrorAction SilentlyContinue
    if ($null -eq $parent) { break }

    if (Test-Path -LiteralPath $LogPath) {
        $file = $null
        $reader = $null
        try {
            $file = [System.IO.File]::Open(
                $LogPath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::ReadWrite
            )
            if ($file.Length -lt $position) { $position = 0L }
            [void]$file.Seek($position, [System.IO.SeekOrigin]::Begin)
            $reader = New-Object System.IO.StreamReader($file, [System.Text.Encoding]::UTF8, $true)
            while (-not $reader.EndOfStream) {
                $line = $reader.ReadLine()
                if ($null -ne $line) { Write-Host $line }
            }
            $position = $file.Position
        } catch {
            # The backend may rotate/reopen the file during a supervised restart.
        } finally {
            if ($null -ne $reader) { $reader.Dispose() }
            elseif ($null -ne $file) { $file.Dispose() }
        }
    }
    Start-Sleep -Milliseconds 250
}

Write-Host "Backend supervisor stopped; detailed log watcher exiting."
