#!/bin/bash
# TAQ pull, job 9 of 4: 35 days, about 143 GB of raw trades (2018-07-17 first). Written by plan_jobs.py.
#$ -N taq_9
#$ -cwd
#$ -j y
#$ -l m_mem_free=15G
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --work /scratch/mit/$USER/taq_work --out /scratch/mit/$USER/taq_out --days-file jobs/days_9.txt
