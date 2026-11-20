# Resume notes (work in progress; delete before the final submission)

Environment: venv `D:\vi-tools\venv` (Python 3.11.9, torch 2.14.1+cu126). Data: `D:\video-indexing\finevideo\data`
(50 files, sizes verified). Weights: `D:\video-indexing\weights\yolo11l.pt`. Run parts: `D:\vi-tools\fullrun\`.

## Full 50-file run: parts

| part | how started | start (PKT) | end | notes |
|---|---|---|---|---|
| (aborted) | after_refs.sh | 11-19 18:19:33 | ~18:25 | CUDA OOM bug, discarded, kept in `fullrun/aborted_oom/` |
| part1 | `D:\vi-tools\fullrun_go.sh part1` | 11-19 18:32:39 | ~18:39 (371 s logged) | agent restart killed it; 9 videos in `.done` |
| part2 | `D:\vi-tools\fullrun_part2.cmd`, detached (WMI) | 11-19 18:41:55 | ~21:57 (11,711 s logged) | laptop shut down (heat); 524 in `.done`, 12 with the timeout error |
| part3 | `D:\vi-tools\fullrun_part3.cmd`, detached (WMI), 8 workers | 11-20 12:09:00 | see `part3.times` | fixed code; 1129 videos (`todo3.txt`), incl. the 12 re-runs |

Each part writes `partN.jsonl`, `partN.jsonl.done` (videos fully written), `partN.log`, `partN.times`.
If a part is interrupted, start the next one DETACHED (not as a child of a Claude shell) with
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c <launcher>.cmd'}`,
running `tools/resume_list.py --done part1.jsonl part2.jsonl ... --redo-errors "timed out waiting for sampling decisions"`.

## When part3.times shows END

```bash
/d/vi-tools/finalize.sh
```

It merges parts 1–3 into `results.jsonl.gz` (`tools/merge_parts.py`: a video is taken from a part only if it is in
that part's `.done`; later parts win, so part3 replaces part2's timeout error lines), then checks:
expect **1650 videos, 461,757 second-lines, ~0 error lines, no "timed out waiting"**; the held-out and tuning
videos inside the full run must score like the validation (held-out F1 0.9905 / exact 96.15% / within-1 99.86%,
tuning 0.9879 / 95.44% / 99.85%) and the held-out lines should equal `val/heldout_final_fp32.jsonl`.
It also prints each part's start/end and last progress line.

Then: fill the `_tbd_` cells in `HARDWARE.md` (total = part1 371 s + part2 ~11,711 s + part3 wall); add the final
numbers to `docs/measurements.md` and README; `git add -f results.jsonl.gz`; delete this file; token scan; push.
`ai_chats/`: copy the agent transcript (tokens replaced by `[REMOVED]`) when told. WRITEUP.md stays for the candidate.
