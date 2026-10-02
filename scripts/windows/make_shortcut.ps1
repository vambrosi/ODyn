# Puts an "ODyn" shortcut on the desktop that opens odyn.bat (next to this
# script) with the ODyn icon. Run once, from this folder:
#
#     powershell -ExecutionPolicy Bypass -File make_shortcut.ps1

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut((Join-Path $desktop "ODyn.lnk"))

$link.TargetPath = Join-Path $here "odyn.bat"
$link.WorkingDirectory = $here
$link.IconLocation = (Join-Path $here "odyn.ico") + ",0"
$link.WindowStyle = 7  # the console starts minimized; the app opens normally
$link.Description = "ODyn entry app (main-sync)"
$link.Save()

Write-Host "Made $(Join-Path $desktop 'ODyn.lnk')"
