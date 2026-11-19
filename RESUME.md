# Resume notes (work in progress; delete before the final submission)

State as of 2026-11-19 evening. Environment: venv `D:\vi-tools\venv` (Python 3.11.9, torch 2.14.1+cu126).
Data: `D:\video-indexing\finevideo\data` (50 files, verified). Weights: `D:\video-indexing\weights\yolo11l.pt`.

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
