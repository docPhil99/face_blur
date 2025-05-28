#!/usr/bin/env bash

#for d in /home/d_phil/intentMAPS/test_set/processed/* ; do
#  echo "$d"
#  python track_and_correct_faces.py --input $d --save --save_path /home/d_phil/intentMAPS/test_set/left_only_processed
#done

for d in /home/d_phil/intentMAPS/processed/real_proc2/* ; do
  if [ -d "$d" ]; then
    echo "$d"
    python track_and_correct_faces.py --input $d --save --save_path /home/d_phil/intentMAPS/processed/left_only_processed
  fi
done