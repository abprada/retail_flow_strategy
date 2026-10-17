"""Download the Robintrack archive and unpack it as is. Nothing here changes the data; cleaning is a later step.

    uv run python scripts/data_acquisition/00_data_robintrack.py

Outputs (gitignored):
    data/raw/robintrack/robintrack-popularity-history.tar.gz   the archive (~529 MB)
    data/raw/robintrack/popularity_export/<TICKER>.csv         its files, byte for byte
"""

from pathlib import Path

from dsr.data import ROBINTRACK_URL, download, extract

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "robintrack"
ARCHIVE = RAW / "robintrack-popularity-history.tar.gz"
FILES = RAW / "popularity_export"


def main() -> None:
    print(f"archive: {ROBINTRACK_URL} -> {ARCHIVE.relative_to(ROOT)} (reused if already downloaded)", flush=True)
    download(ROBINTRACK_URL, ARCHIVE)
    print(f"unpacking -> {FILES.relative_to(ROOT)} (skipped if already unpacked)", flush=True)
    extract(ARCHIVE, FILES, "tmp/popularity_export")
    csvs = list(FILES.iterdir())
    print(f"{len(csvs):,} files, {sum(p.stat().st_size for p in csvs) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
