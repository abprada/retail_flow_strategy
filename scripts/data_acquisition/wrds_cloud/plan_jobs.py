"""Split the full TAQ pull into SGE jobs of equal size, and write their job scripts into the WRDS Cloud bundle.

    uv run python scripts/data_acquisition/wrds_cloud/plan_jobs.py [--jobs 5] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                                                  [--days-file days.txt --first-job 6] [--prices]

WRDS Cloud allows a user 10 batch slots and 150 GB of memory at once (explanations/taq/wrds_cloud.md); a job takes 2
slots, so 5 jobs of 30 GB (m_mem_free is per slot: 15 GB x 2) use both limits fully. Days are weighted by the size of
their raw trades (the typical day of their year, from the coverage check and the pilot), so each job gets about the
same amount of work, not the same number of days. Within every job the days run in order of priority, so the most
useful years arrive first: 2018-05 to 2020-08 (Robintrack, Model 1), then 2020-09 to 2025, then 2014 to 2018-04.

With --from and --to, only the days in that range; with --days-file, only the days listed (one YYYY-MM-DD per line):
for pulling some days again after a change to the query. With --first-job k > 1 the jobs are numbered from k, the
existing job files are kept, and the submit script is submit_<k>_<last>.sh, so jobs still running are not touched.

Writes, into scripts/data_acquisition/wrds_cloud/jobs/ (and the bundle, when it is rebuilt):
    days_<k>.txt     the days of job k, in the order it runs them
    job_<k>.sh       the SGE job script of job k
    submit_all.sh    submits every job (run from ~/taq on WRDS Cloud: bash jobs/submit_all.sh)

With --prices, the jobs of the price-only pull (taq_cloud_day.py --prices, output in /scratch/mit/$USER/taq_prices)
instead, written next to the others without touching them: prices_days_<k>.txt, prices_<k>.sh and submit_prices.sh.
Their days run in date order, the training days (to 2023) first, then the test days.
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

PRICES_JOB = """#!/bin/bash
# TAQ price-only pull, job {k} of {n}: {days} days ({first} first). Written by plan_jobs.py --prices.
#$ -N taq_p{k}
#$ -cwd
#$ -j y
#$ -l m_mem_free={mem}G
#$ -o /scratch/mit/$USER/taq_logs/$JOB_NAME.$JOB_ID.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd ~/taq
python3 taq_cloud_day.py --prices --work /scratch/mit/$USER/taq_work_prices --out /scratch/mit/$USER/taq_prices \\
    --days-file jobs/prices_days_{k}.txt
"""
PRICES_SECONDS_PER_GB = 12  # of the full pull's raw trades: the price-only pull copies about half and computes little

SUBMIT = """#!/bin/bash
# Submit jobs {ks} of the TAQ pull. Run from ~/taq on WRDS Cloud (qsub fails on absolute paths there).
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
    parser.add_argument("--from", dest="start", default="0000-00-00")
    parser.add_argument("--to", dest="end", default="9999-99-99")
    parser.add_argument("--days-file", type=Path)
    parser.add_argument("--first-job", type=int, default=1)
    parser.add_argument("--prices", action="store_true", help="jobs of the price-only pull")
    args = parser.parse_args()
    n, first = args.jobs, args.first_job

    with open(HERE / "days.csv") as f:
        days = [r["date"] for r in csv.DictReader(f) if args.start <= r["date"] <= args.end]
    if args.days_file:
        listed = set(args.days_file.read_text().split())
        if listed - set(days):
            raise SystemExit(f"not trading days in range: {sorted(listed - set(days))[:5]}")
        days = [d for d in days if d in listed]
    order = sorted(days, key=lambda d: (d > "2023-12-31", d) if args.prices else (priority(d), d))
    jobs: list[list[str]] = [[] for _ in range(n)]
    load = [0.0] * n
    for day in order:  # each day to the job with the least work so far
        k = load.index(min(load))
        jobs[k].append(day)
        load[k] += GB_PER_DAY[day[:4]]

    OUT.mkdir(exist_ok=True)
    if args.prices:
        for old in OUT.glob("prices*"):
            old.unlink()
        for k, (job, gb) in enumerate(zip(jobs, load), start=1):
            (OUT / f"prices_days_{k}.txt").write_text("\n".join(job) + "\n")
            (OUT / f"prices_{k}.sh").write_text(PRICES_JOB.format(k=k, n=n, days=len(job), first=job[0],
                                                                  mem=MEMORY_PER_SLOT_GB))
            print(f"job {k}: {len(job):,} days, ~{gb * PRICES_SECONDS_PER_GB / 3600:.1f} hours")
        (OUT / "submit_prices.sh").write_text(SUBMIT.format(ks=" ".join(str(k) for k in range(1, n + 1)))
                                              .replace("jobs/job_$k.sh", "jobs/prices_$k.sh")
                                              .replace("Submit jobs", "Submit the price-only jobs"))
        print(f"all: {len(days):,} days; with {n} jobs at once about {max(load) * PRICES_SECONDS_PER_GB / 3600:.1f} hours")
        return
    if first == 1:
        for old in OUT.glob("*"):
            if not old.name.startswith("prices"):
                old.unlink()
    for k, (job, gb) in enumerate(zip(jobs, load), start=first):
        (OUT / f"days_{k}.txt").write_text("\n".join(job) + "\n")
        (OUT / f"job_{k}.sh").write_text(JOB.format(k=k, n=n, days=len(job), gb=gb, first=job[0], mem=MEMORY_PER_SLOT_GB))
        print(f"job {k}: {len(job):,} days, ~{gb:,.0f} GB of raw trades, ~{gb * SECONDS_PER_GB / 3600:.1f} hours")
    submit = "submit_all.sh" if first == 1 else f"submit_{first}_{first + n - 1}.sh"
    (OUT / submit).write_text(SUBMIT.format(ks=" ".join(str(k) for k in range(first, first + n))))
    total = sum(load)
    print(f"all: {len(days):,} days, ~{total / 1000:.1f} TB of raw trades; with {n} jobs at once about "
          f"{max(load) * SECONDS_PER_GB / 3600:.1f} hours")


if __name__ == "__main__":
    main()
