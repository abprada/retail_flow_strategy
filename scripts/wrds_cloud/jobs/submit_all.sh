#!/bin/bash
# Submit every job of the TAQ pull. Run from ~/taq (qsub fails on absolute paths on WRDS): bash jobs/submit_all.sh
mkdir -p /scratch/mit/$USER/taq_logs
for k in 1 2 3 4 5; do qsub jobs/job_$k.sh; done
qstat
