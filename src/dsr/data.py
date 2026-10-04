"""Raw data ingestion: download, cache, and point-in-time alignment.

Every record must carry the timestamp at which it became publicly available,
not just the event date it describes.
"""

import getpass
import os
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import psycopg

USER_AGENT = "dsr-research/0.1"

# Robintrack: hourly Robinhood holder counts, one CSV per ticker under tmp/popularity_export/ in the archive.
ROBINTRACK_URL = "https://robintrack-data.ameo.design/robintrack-popularity-history.tar.gz"

# WRDS PostgreSQL. The `wrds` package pins pandas < 2.3, so we connect with psycopg directly.
WRDS_HOST, WRDS_PORT, WRDS_DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"


def download(url: str, dest: Path, user_agent: str = USER_AGENT) -> Path:
    """Fetch url to dest once. A partial transfer stays in dest.part, never at dest."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urlopen(Request(url, headers={"User-Agent": user_agent})) as resp, open(part, "wb") as f:
        expected = resp.headers.get("Content-Length")
        shutil.copyfileobj(resp, f, length=1 << 20)
    got = part.stat().st_size
    if expected is not None and got != int(expected):
        raise OSError(f"{url}: got {got} bytes, expected {expected}")
    part.rename(dest)
    return dest


def extract(archive: Path, dest: Path, member_dir: str) -> Path:
    """Unpack the directory member_dir of a tar archive into dest, files byte for byte. Skipped if dest exists."""
    dest = Path(dest)
    if dest.exists():
        return dest
    staging = dest.with_name(dest.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")
    (staging / member_dir).rename(dest)
    shutil.rmtree(staging)
    return dest


def wrds_connect(username: str) -> psycopg.Connection:
    """Open a WRDS connection. The password comes from ~/.pgpass or PGPASSWORD, else an interactive prompt.

    WRDS holds the login until its Duo push is approved; after 120 s without approval the connection fails.
    """
    password = None
    if not (Path.home() / ".pgpass").exists() and "PGPASSWORD" not in os.environ:
        password = getpass.getpass(f"WRDS password for {username}: ")
    return psycopg.connect(host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB, user=username, password=password,
                           sslmode="require", connect_timeout=120)


def copy_query(conn, query: str, dest: Path) -> Path:
    """Write a query's result to dest as Postgres CSV with a header, unchanged. Skipped if dest exists."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with conn.cursor() as cur, cur.copy(f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER)") as copy, \
                open(part, "wb") as f:
            f.writelines(copy)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.rename(dest)
    return dest
