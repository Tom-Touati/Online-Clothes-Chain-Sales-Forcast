# Run one Python job at below-normal priority with at most 3 threads (user, 18 Nov: "you can use the laptop but don't let it lag the computer").
# Usage: powershell -File pipeline/lowrun.ps1 <log file> <script> [args...]
param([string]$Log, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$env:OMP_NUM_THREADS = "3"; $env:MKL_NUM_THREADS = "3"; $env:OPENBLAS_NUM_THREADS = "3"; $env:NUMEXPR_NUM_THREADS = "3"; $env:NJOBS = "3"
$py = "C:\Users\tomto\workspaces\edikted\.venv\Scripts\python.exe"
$p = Start-Process -FilePath $py -ArgumentList $Rest -WorkingDirectory "C:\Users\tomto\workspaces\edikted" -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -NoNewWindow -PassThru
Start-Sleep -Milliseconds 500
try { $p.PriorityClass = 'BelowNormal' } catch {}
Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | ForEach-Object { try { (Get-Process -Id $_.ProcessId).PriorityClass = 'BelowNormal' } catch {} }
$p.WaitForExit(); "exit $($p.ExitCode)"
