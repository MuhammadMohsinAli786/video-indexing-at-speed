# Measurements log

Facts and numbers only. Each entry has a date (PKT). Used by the candidate to write `WRITEUP.md`.

## Dev machine

- CPU: Intel Core i7-8850H @ 2.60 GHz, 6 cores / 12 threads
- RAM: 16 GB
- GPU: NVIDIA Quadro P2000, 4 GB (Pascal, sm_61), driver 573.71 (CUDA 12.8)
- Disk: Samsung MZVLB512HAJQ NVMe SSD
- OS: Windows 11 Pro (10.0.26300), Python 3.11.9

## Log

- 2026-11-19: environment set up. PyPI throughput on the dev network was 4-250 KB/s (Fastly edge dependent); wheels fetched with parallel resumable curl pinned to the fastest edge IP. HF download ~6 MB/s per connection.
