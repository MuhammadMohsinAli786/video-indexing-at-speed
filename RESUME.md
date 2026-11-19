# Resume notes (work in progress; delete before the final submission)

Environment: venv `D:\vi-tools\venv` (Python 3.11.9, torch 2.14.1+cu126). Data: `D:\video-indexing\finevideo\data`
(50 files, sizes verified). Weights: `D:\video-indexing\weights\yolo11l.pt`. Run parts: `D:\vi-tools\fullrun\`.

## Full 50-file run: state (2026-11-19 evening)

| part | how started | start (PKT) | end | notes |
|---|---|---|---|---|
| (aborted) | after_refs.sh | 18:19:33 | ~18:25 | CUDA OOM bug, discarded, kept in `fullrun/aborted_oom/` |
| part1 | `D:\vi-tools\fullrun_go.sh part1` | 18:32:39 | ~18:39 (killed, 371 s logged) | Claude Code restart killed it; 9 videos in `part1.jsonl.done` |
| part2 | `D:\vi-tools\fullrun_part2.cmd`, detached via WMI | 18:41:55 | see `part2.times` | resumes the other 1641 videos (`todo2.txt`) |

Each part writes `partN.jsonl`, `partN.jsonl.done` (videos fully written), `partN.log`, `partN.times`.

## If part2 is interrupted too

Launch the next part DETACHED (not as a child of a Claude shell), e.g. a `fullrun_part3.cmd` like part2's with
`--done D:/vi-tools/fullrun/part1.jsonl D:/vi-tools/fullrun/part2.jsonl` for `resume_list.py`, started via
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c D:\vi-tools\fullrun_part3.cmd'}`.

## When the last part has finished

```bash
cd /d/video-indexing
/d/vi-tools/venv/Scripts/python.exe tools/merge_parts.py --input finevideo/data \
  --parts /d/vi-tools/fullrun/part1.jsonl /d/vi-tools/fullrun/part2.jsonl --out results.jsonl.gz
```

`merge_parts.py` takes a video from a part only if it is listed in that part's `.done` file, so part1's partial
last video is dropped automatically. Expect 1650 videos written, 0 missing.

Then: full-run time in `HARDWARE.md` = part1 wall (371 s + shutdown) + part2 wall (from `part2.times`/log),
excluding the gap and the aborted attempt; README final numbers; token scan; push; delete this file.
`ai_chats/` is added by the main session; WRITEUP.md stays for the candidate.
