#!/bin/bash
# Submit jobs 6 7 8 9 of the TAQ pull. Run from ~/taq on WRDS Cloud (qsub fails on absolute paths there).
mkdir -p /scratch/mit/$USER/taq_logs
for k in 6 7 8 9; do qsub jobs/job_$k.sh; done
qstat
