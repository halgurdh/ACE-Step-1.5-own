$ErrorActionPreference = "Continue"
$env:PYTHONFAULTHANDLER = "1"
"Wrapper started at $(Get-Date)" | Out-File -FilePath output\all_genres_run\exitcode4.txt
$proc = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-X faulthandler generate_track.py --quality best --guidance-scale 7.0 --duration 150 --all-genres --amount 20 --format mp3 --no-adg --no-offload-dit --lm-cfg-scale 3.0 --target-lufs -14 --enable-apollo-restoration --enable-audiosr-upscale --audiosr-ddim-steps 25 --candidates 2 --output-dir output/all_genres_run" -RedirectStandardOutput output\all_genres_run\stdout4.log -RedirectStandardError output\all_genres_run\stderr4.log -NoNewWindow -PassThru -Wait
"$($proc.ExitCode)" | Out-File -FilePath output\all_genres_run\exitcode4.txt -Append
"Process exited at $(Get-Date) with code $($proc.ExitCode)" | Out-File -FilePath output\all_genres_run\exitcode4.txt -Append
