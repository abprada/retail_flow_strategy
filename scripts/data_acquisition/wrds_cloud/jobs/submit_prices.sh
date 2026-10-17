#!/bin/bash
# Submit the price-only jobs 1 2 3 4 5 of the TAQ pull. Run from ~/taq on WRDS Cloud (qsub fails on absolute paths there).
mkdir -p /scratch/mit/$USER/taq_logs
for k in 1 2 3 4 5; do qsub jobs/prices_$k.sh; done
qstat
