#!/bin/bash
# TAQ pull, job 4 of 5: 603 days, about 2,005 GB of raw trades (2018-05-04 first). Written by plan_jobs.py.
#$ -N taq_4
#$ -cwd
#$ -j y
#$ -l m_mem_free=15G
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --work /scratch/mit/$USER/taq_work --out /scratch/mit/$USER/taq_out --days-file jobs/days_4.txt
