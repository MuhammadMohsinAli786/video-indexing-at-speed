# Write-up

Answers written by the candidate (Muhammad Mohsin Ali), without AI.
Facts and numbers to draw on are in `docs/measurements.md`.

## 1. Timings: first vs final version on the full 50 files, and the single biggest saving
  Every fram running required 4255 seconds  and 23 held videos, My final run take 287s and its 14.8x faster. taking every fame from ultralytics api is too much slow expected 360 hours and by usig our pipeline alnost 150 hours then on out method that is described The full 50 files run complete in 9 hours and 38 minutes. The biggest saving is 1 fram per second and for unsure seens 3 frames per seconds. in this 5.8% frames analyse only.

## 2. The pipeline, step by step 
 its to read the mp4 from parquet, then 8 workers decode the frames, then every seconds each frame  on yolov11l on gpu. if confidence is between 0.25 or 0.6 and differ from the neighbouring seconds then 3 frames scnerio implemented.every seconds peoples and objects are counted in result file.

## 3. Where the time goes, and how it was measured
 first decode on cpu and gpu takes equal time. The time is measured inside the measurement.md  by time breakdown. and gpu is  busy more then 97% all the time.



## 4. How correctness was checked after speeding up 
 the every frame ans results F1= 0.9905, exact = 96.15%, within = 99.86% with 23 held videos not used inside the tunning. and 180 frames results are equal to the ultralytics.


## 5. One AI suggestion that was wrong or rejected, and why 

The ai tried the batch size 16 it full the 4 GB gpu memory,that's why first run is to stopped after 5 minutes. then batch size is accordign to gpu memory.

## 6. Next steps with one more week 

There are some frames have more rush of peoples increase the frames and take the decode to the gpu and try other gpu.if i have one more week i add more step that normaly 3 frame and on required 9 frames so that its increase the exact values in the videos that have more rush per seconds and some other way i'll used for it.
