"""Download the TAQ pull's finished days from WRDS Cloud to this computer, check them, and free their space on WRDS.

    uv run python scripts/wrds_cloud/download.py [--days-file days.txt] [--streams 4] [--every 10] [--delete-remote] [--once]

Needs one SSH connection that later commands share, so that WRDS asks for the password and Duo only once. Add to
~/.ssh/config on this computer:

    Host wrds
        HostName wrds-cloud.wharton.upenn.edu
        User abprada
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h-%p
        ControlPersist 24h

then open it once with `ssh -fN wrds` (password + Duo). Every `ssh wrds` and `rsync` after that reuses it, for 24 hours.

Each round (every --every minutes until every day asked for is here, made by the current code version, or once with
--once; the days asked for are every trading day, or those in --days-file, one YYYY-MM-DD per line):
  1. lists the finished days on WRDS (folders YYYYMMDD in the output folder; .part folders are still being written) and
     copies the jobs' manifests;
  2. downloads the days not yet here from the current code version with rsync, in --streams transfers at once (a day
     pulled again replaces the one here);
  3. checks each new day: the four tables open, the conservation checks pass (src/dsr/taq.py), and the row counts
     equal those the job recorded in its manifest; a day that fails is downloaded again next round;
  4. records the checked days and their code version in downloaded.csv and, with --delete-remote, deletes them from
     WRDS scratch (500 GB
     shared by the institution, emptied after 7 days), only after the check passed.

Output (gitignored): data/raw/taq/pull/<YYYYMMDD>/ (one folder per WCT day, as the job wrote it),
    manifests/ (the jobs' manifests), downloaded.csv (the checked days: day, version, checked_at).
"""

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dsr.taq import KEYS, check_day

_spec = importlib.util.spec_from_file_location("taq_cloud_day", Path(__file__).resolve().parent / "taq_cloud_day.py")
_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_job)
VERSION = _job.code_version()  # only days made by the current code are downloaded

LOCAL = ROOT / "data" / "raw" / "taq" / "pull"
DAYS = Path(__file__).resolve().parent / "days.csv"


def ssh(host: str, command: str) -> str:
    return subprocess.run(["ssh", host, command], check=True, capture_output=True, text=True).stdout


def rsync(host: str, remote: str, names: list[str], dest: Path) -> int:
    """Copy the folders `names` of `remote` into dest; returns rsync's exit code."""
    listing = dest / f".rsync_{names[0]}.txt"
    listing.write_text("\n".join(names) + "\n")
    code = subprocess.run(["rsync", "-a", "-r", "--partial", f"--files-from={listing}", f"{host}:{remote}/", f"{dest}/"],
                          check=False).returncode
    listing.unlink()
    return code


def manifest_rows(folder: Path) -> dict[str, dict]:
    """The manifest line of every day made by the current code version whose checks passed, from all jobs' manifests."""
    rows = {}
    for path in sorted(folder.glob("manifest*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["checks_passed"] == "True" and row["version"] == VERSION:
                    rows[row["day"].replace("-", "")] = row
    return rows


def check(ymd: str, close: str, expected: dict | None) -> str | None:
    """None if the downloaded day is complete and consistent, else what is wrong."""
    folder = LOCAL / ymd
    try:
        tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in KEYS}
    except (OSError, ValueError) as e:
        return f"cannot read: {e}"
    hours, minutes = map(int, close.split(":"))
    regular_bars = ((hours * 60 + minutes - 10) - (9 * 60 + 30)) // 10
    checks = check_day(tables["daily"], tables["cells"], tables["bars"], tables["audit"], regular_bars)
    if not checks["passed"].all():
        return "conservation checks fail: " + "; ".join(checks.loc[~checks["passed"], "check"])
    if expected is None:
        return "not in any manifest"
    wrong = [name for name in KEYS if len(tables[name]) != int(expected[f"{name}_rows"])]
    return f"row counts differ from the manifest: {wrong}" if wrong else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="wrds", help="the SSH host alias (see above)")
    parser.add_argument("--days-file", type=Path, help="one YYYY-MM-DD per line (default: every trading day)")
    parser.add_argument("--remote", default="/scratch/mit/abprada/taq_out")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--every", type=float, default=10, help="minutes between rounds")
    parser.add_argument("--delete-remote", action="store_true", help="delete checked days from WRDS scratch")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if subprocess.run(["ssh", "-O", "check", args.host], capture_output=True, check=False).returncode != 0:
        raise SystemExit(f"no shared SSH connection to {args.host}: run `ssh -fN {args.host}` first (password + Duo)")
    with open(DAYS) as f:
        closes = {r["date"].replace("-", ""): r["close"] for r in csv.DictReader(f)}
    wanted = set(closes)
    if args.days_file:
        wanted = {d.strip().replace("-", "") for d in args.days_file.read_text().split()}
        if wanted - set(closes):
            raise SystemExit(f"not trading days: {sorted(wanted - set(closes))[:5]}")
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / "manifests").mkdir(exist_ok=True)
    log = LOCAL / "downloaded.csv"
    logged = pd.read_csv(log, dtype=str) if log.exists() else pd.DataFrame(columns=["day", "version"])
    done = set(logged.loc[logged["version"] == VERSION, "day"]) & wanted
    print(f"code version {VERSION}; {len(done):,} of {len(wanted):,} days already here", flush=True)

    while True:
        start = time.perf_counter()
        listing = ssh(args.host, f"ls -1 {args.remote}").split()
        finished = sorted(n for n in listing if re.fullmatch(r"\d{8}", n) and n in wanted)
        manifests = [n for n in listing if n.startswith("manifest") and n.endswith(".csv")]
        if manifests:
            rsync(args.host, args.remote, manifests, LOCAL / "manifests")
        expected = manifest_rows(LOCAL / "manifests")
        new = [d for d in finished if d not in done and d in expected]
        if new:
            batches = [new[i::args.streams] for i in range(args.streams) if new[i::args.streams]]
            with ThreadPoolExecutor(len(batches)) as pool:
                codes = list(pool.map(lambda b: rsync(args.host, args.remote, b, LOCAL), batches))
            if any(codes):
                print(f"rsync exit codes {codes}; failed days are retried next round", flush=True)
            checked, problems = [], {}
            for ymd in new:
                problem = check(ymd, closes[ymd], expected.get(ymd))
                if problem:
                    problems[ymd] = problem
                else:
                    checked.append(ymd)
            if checked:
                pd.DataFrame({"day": checked, "version": VERSION,
                              "checked_at": pd.Timestamp.now().isoformat(timespec="seconds")}).to_csv(
                    log, mode="a", header=not log.exists(), index=False)
                done |= set(checked)
                if args.delete_remote:
                    for i in range(0, len(checked), 200):
                        ssh(args.host, "rm -rf " + " ".join(f"{args.remote}/{d}" for d in checked[i:i + 200]))
            for ymd, problem in problems.items():
                print(f"{ymd}: {problem}; downloaded again next round", flush=True)
            mb = sum(p.stat().st_size for d in checked for p in (LOCAL / d).glob("*.parquet")) / 1e6
            seconds = time.perf_counter() - start
            print(f"{time.strftime('%H:%M')}: {len(checked)} days downloaded and checked ({mb:,.0f} MB in {seconds:.0f} s, "
                  f"{mb / max(seconds, 1):.1f} MB/s){', deleted from WRDS' if args.delete_remote and checked else ''}; "
                  f"{len(done):,} of {len(wanted):,} days here", flush=True)
        else:
            print(f"{time.strftime('%H:%M')}: nothing new; {len(done):,} of {len(wanted):,} days here", flush=True)
        if args.once or len(done) >= len(wanted):
            break
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
