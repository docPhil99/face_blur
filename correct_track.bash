#!/usr/bin/env bash

#for d in /home/d_phil/intentMAPS/test_set/processed/* ; do
#  echo "$d"
#  python track_and_correct_faces.py --input $d --save --save_path /home/d_phil/intentMAPS/test_set/left_only_processed
#done
#
#for d in /home/d_phil/intentMAPS/processed/real_proc2/* ; do
#  if [ -d "$d" ]; then
#    echo "$d"
#    python track_and_correct_faces.py --input $d --save --save_path /home/d_phil/intentMAPS/processed/left_only_processed
#  fi
#done



for d in /home/d_phil/intentMAPS/processed/complete_corrected/27_03_2025/* ; do
  if [ -d "$d" ]; then
    echo "$d"
    python track_and_correct_faces.py --input $d --save --no_display --save_path /home/d_phil/intentMAPS/processed/left_only_processed/27_03_2025
  fi
done
echo "Day 1"
for d in /home/d_phil/intentMAPS/processed/complete_corrected/25_03_2025/* ; do
  if [ -d "$d" ]; then
    echo "$d"
    python track_and_correct_faces.py --input $d --save  --no_display --save_path /home/d_phil/intentMAPS/processed/left_only_processed/25_03_2025
  fi
done

