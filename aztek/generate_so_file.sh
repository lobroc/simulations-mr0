#!/bin/sh

gcc -c -fPIC AZTEK/AZTEK_trajectory_comp.c -o aztek_traj.o
gcc -shared -Wl,-soname,aztek_traj.so -o libaztek.so -o libaztek.so aztek_traj.o
rm aztek_traj.o
