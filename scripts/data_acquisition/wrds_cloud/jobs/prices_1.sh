#!/bin/bash
# TAQ price-only pull, job 1 of 5: 604 days (2013-12-31 first). Written by plan_jobs.py --prices.
#$ -N taq_p1
#$ -cwd
#$ -j y
#$ -l m_mem_free=15G
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --prices --work /scratch/mit/$USER/taq_work_prices --out /scratch/mit/$USER/taq_prices \
    --days-file jobs/prices_days_1.txt
