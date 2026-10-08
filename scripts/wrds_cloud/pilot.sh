#!/bin/bash
# The pilot of the TAQ pull (explanations/taq_pull.md, section 8): ten days chosen to break the query, as one SGE job.
#   mkdir -p /scratch/mit/$USER/taq_logs && cd ~/taq && qsub pilot.sh   (qsub ~/taq/pilot.sh fails: WRDS doubles the path)
#   qstat                                          # is it running?
#   tail -f /scratch/mit/$USER/taq_logs/taq_pilot.*.log
#$ -N taq_pilot
#$ -cwd
#$ -j y
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --work /scratch/mit/$USER/taq_work_pilot --out /scratch/mit/$USER/taq_out \
    2013-12-31 2014-01-02 2015-06-01 2015-07-08 2016-10-03 2016-10-17 2019-11-29 2020-03-16 2025-04-07 2025-06-02
