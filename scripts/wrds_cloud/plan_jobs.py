"""Split the full TAQ pull into SGE jobs of equal size, and write their job scripts into the WRDS Cloud bundle.

    uv run python scripts/wrds_cloud/plan_jobs.py [--jobs 5]

WRDS Cloud allows a user 10 batch slots and 150 GB of memory at once (explanations/TAQ/wrds_cloud.md); a job takes 2
slots, so 5 jobs of 30 GB (m_mem_free is per slot: 15 GB x 2) use both limits fully. Days are weighted by the size of
their raw trades (the typical day of their year, from the coverage check and the pilot), so each job gets about the
same amount of work, not the same number of days. Within every job the days run in order of priority, so the most
useful years arrive first: 2018-05 to 2020-08 (Robintrack, Model 1), then 2020-09 to 2025, then 2014 to 2018-04.

Writes, into scripts/wrds_cloud/jobs/ (and the bundle, when it is rebuilt):
    days_<k>.txt     the days of job k, in the order it runs them
    job_<k>.sh       the SGE job script of job k
    submit_all.sh    submits every job (run from ~/taq on WRDS Cloud: bash jobs/submit_all.sh)
"""

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "jobs"

# raw trades of a typical day, GB of CSV: the coverage check's June day of each year at about 60 bytes a trade, and
# the pilot's measured days; only used to balance the jobs
GB_PER_DAY = {"2013": 1.0, "2014": 1.5, "2015": 1.8, "2016": 2.0, "2017": 2.0, "2018": 2.1, "2019": 2.9, "2020": 3.3,
              "2021": 4.1, "2022": 4.8, "2023": 4.7, "2024": 4.8, "2025": 5.9}
SECONDS_PER_GB = 25  # with the pull of one day overlapping the computing of the previous one (about 42 s/GB without)
MEMORY_PER_SLOT_GB = 15  # x 2 slots = 30 GB a job; 5 jobs = 150 GB, the per-user limit

JOB = """#!/bin/bash
# TAQ pull, job {k} of {n}: {days} days, about {gb:,.0f} GB of raw trades ({first} first). Written by plan_jobs.py.
#$ -N taq_{k}
#$ -cwd
#$ -j y
#$ -l m_mem_free={mem}G
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --work /scratch/mit/$USER/taq_work --out /scratch/mit/$USER/taq_out --days-file jobs/days_{k}.txt
"""

SUBMIT = """#!/bin/bash
# Submit every job of the TAQ pull. Run from ~/taq (qsub fails on absolute paths on WRDS): bash jobs/submit_all.sh
mkdir -p /scratch/mit/$USER/taq_logs
for k in {ks}; do qsub jobs/job_$k.sh; done
qstat
"""


def priority(day: str) -> int:
    if "2018-05-01" <= day <= "2020-08-31":
        return 0
    if day > "2020-08-31":
        return 1
    return 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=5)
    n = parser.parse_args().jobs

    with open(HERE / "days.csv") as f:
        days = [r["date"] for r in csv.DictReader(f)]
    order = sorted(days, key=lambda d: (priority(d), d))
    jobs: list[list[str]] = [[] for _ in range(n)]
    load = [0.0] * n
    for day in order:  # each day to the job with the least work so far
        k = load.index(min(load))
        jobs[k].append(day)
        load[k] += GB_PER_DAY[day[:4]]

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    for k, (job, gb) in enumerate(zip(jobs, load), start=1):
        (OUT / f"days_{k}.txt").write_text("\n".join(job) + "\n")
        (OUT / f"job_{k}.sh").write_text(JOB.format(k=k, n=n, days=len(job), gb=gb, first=job[0], mem=MEMORY_PER_SLOT_GB))
        print(f"job {k}: {len(job):,} days, ~{gb:,.0f} GB of raw trades, ~{gb * SECONDS_PER_GB / 3600:.1f} hours")
    (OUT / "submit_all.sh").write_text(SUBMIT.format(ks=" ".join(str(k) for k in range(1, n + 1))))
    total = sum(load)
    print(f"all: {len(days):,} days, ~{total / 1000:.1f} TB of raw trades; with {n} jobs at once about "
          f"{max(load) * SECONDS_PER_GB / 3600:.1f} hours")


if __name__ == "__main__":
    main()
