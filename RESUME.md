# Resume notes (work in progress; delete before the final submission)

State as of 2026-11-19 evening. Environment: venv `D:\vi-tools\venv` (Python 3.11.9, torch 2.14.1+cu126).
Data: `D:\video-indexing\finevideo\data` (50 files, verified). Weights: `D:\video-indexing\weights\yolo11l.pt`.

## Automated chain (started 2026-11-19 ~17:00)

`D:\vi-tools\after_refs.sh` waits for the held-out reference (`val/ref_heldout.jsonl`), then runs the final
settings on held-out (FP32 + FP16) and on the tuning set, scores them into `D:\vi-tools\logs\after_refs.log`,
and starts the full run (`D:\vi-tools\fullrun\part1.jsonl` + `.done`, log `part1.log`) if held-out FP32 keeps
F1 >= 0.98, exact >= 92%, within-1 >= 98%. Check `after_refs.log` first.

## Full 50-file run (resumable)

Run parts go to `D:\vi-tools\fullrun\` (outside the repo). Each part has a `.done` list of finished videos.

Start (first part):

```bash
cd /d/video-indexing
/d/vi-tools/venv/Scripts/python.exe run.py --input finevideo/data --output /d/vi-tools/fullrun/part1.jsonl \
  --done-file /d/vi-tools/fullrun/part1.jsonl.done  > /d/vi-tools/fullrun/part1.log 2>&1
```

Resume after an interruption (part N+1):

```bash
cd /d/video-indexing
/d/vi-tools/venv/Scripts/python.exe tools/resume_list.py --input finevideo/data \
  --done /d/vi-tools/fullrun/part*.jsonl --out /d/vi-tools/fullrun/todo.txt
/d/vi-tools/venv/Scripts/python.exe run.py --input finevideo/data --output /d/vi-tools/fullrun/partN.jsonl \
  --done-file /d/vi-tools/fullrun/partN.jsonl.done --videos /d/vi-tools/fullrun/todo.txt > /d/vi-tools/fullrun/partN.log 2>&1
```

Merge when nothing is left to do:

```bash
/d/vi-tools/venv/Scripts/python.exe tools/merge_parts.py --input finevideo/data \
  --parts /d/vi-tools/fullrun/part1.jsonl /d/vi-tools/fullrun/part2.jsonl ... --out results.jsonl.gz
```

Full run time for HARDWARE.md = sum of the `seconds` in each part log's final `done: {...}` line
(plus the wall-clock start/end times of each part).

## Next steps

See the bottom of `docs/measurements.md` for the latest numbers. Remaining after the full run:
results.jsonl.gz + HARDWARE.md time, README final numbers, final repo check (token scan), `ai_chats/` is added
by the main session, WRITEUP.md is for the candidate.
