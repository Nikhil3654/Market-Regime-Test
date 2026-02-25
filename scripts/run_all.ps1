param(
  [string]$VenvPath = ".\.venv\Scripts\Activate.ps1"
)

Write-Host "Activating venv: $VenvPath"
. $VenvPath

Write-Host "Running dataset build (if needed)"
python scripts\build_model_dataset.py

Write-Host "Running Day4 baselines + backtest"
python scripts\run_day4.py

Write-Host "Running Day5 plots"
python scripts\make_day5_plots.py

Write-Host "Done. Outputs in outputs/day4 and outputs/day5"