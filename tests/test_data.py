import io
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dsr.data import copy_query, download, extract


def _tar(path, files: dict[str, bytes]):
    with tarfile.open(path, "w:gz") as tar:
        d = tarfile.TarInfo("tmp/popularity_export/")
        d.type = tarfile.DIRTYPE
        tar.addfile(d)
        for name, data in files.items():
            info = tarfile.TarInfo(f"tmp/popularity_export/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_extract_unpacks_files_unchanged(tmp_path):
    files = {
        "AAA.csv": b'timestamp,users_holding\n"2018-05-02 04:54:12",699\n',
        "_OUT.csv": b'timestamp,users_holding\n"2019-01-02 15:00:00",1385.0\n',   # quirks stay as they are
    }
    archive = _tar(tmp_path / "rt.tar.gz", files)
    dest = tmp_path / "raw" / "popularity_export"

    assert extract(archive, dest, "tmp/popularity_export") == dest

    assert sorted(p.name for p in dest.iterdir()) == ["AAA.csv", "_OUT.csv"]
    for name, data in files.items():
        assert (dest / name).read_bytes() == data
    assert not dest.with_name(dest.name + ".part").exists()


def test_extract_skips_existing_destination(tmp_path):
    archive = _tar(tmp_path / "rt.tar.gz", {"AAA.csv": b"new"})
    dest = tmp_path / "popularity_export"
    dest.mkdir()
    (dest / "AAA.csv").write_bytes(b"old")
    extract(archive, dest, "tmp/popularity_export")
    assert (dest / "AAA.csv").read_bytes() == b"old"


def test_download_fetches_once(tmp_path):
    served = tmp_path / "srv"
    served.mkdir()
    (served / "blob.bin").write_bytes(b"x" * 5000)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(served)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/blob.bin"
    dest = tmp_path / "raw" / "blob.bin"
    try:
        assert download(url, dest) == dest
        assert dest.read_bytes() == b"x" * 5000
        assert not dest.with_name(dest.name + ".part").exists()
    finally:
        server.shutdown()
    # a complete copy is never fetched again, even with the server gone
    assert download(url, dest) == dest


class _FakeConn:
    """Stands in for a psycopg connection: cursor().copy(sql) yields the given chunks."""

    def __init__(self, chunks):
        self.chunks, self.sql = chunks, None

    def cursor(self):
        return self

    def copy(self, sql):
        self.sql = sql
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self.chunks


def test_copy_query_writes_postgres_csv_unchanged(tmp_path):
    chunks = [b"permno,ticker,namedt\n", b"14593,AAPL,1980-12-12\n", b"10107,MSFT,1986-03-13\n"]
    conn = _FakeConn(chunks)
    dest = tmp_path / "raw" / "crsp" / "stocknames.csv"

    assert copy_query(conn, "SELECT * FROM crsp.stocknames", dest) == dest

    assert dest.read_bytes() == b"".join(chunks)
    assert conn.sql == "COPY (SELECT * FROM crsp.stocknames) TO STDOUT WITH (FORMAT csv, HEADER)"
    assert not dest.with_name(dest.name + ".part").exists()


def test_copy_query_skips_existing_and_leaves_nothing_on_failure(tmp_path):
    dest = tmp_path / "stocknames.csv"
    dest.write_bytes(b"old")
    assert copy_query(None, "SELECT 1", dest) == dest          # never touches the connection
    assert dest.read_bytes() == b"old"

    def _broken():
        yield b"permno\n"
        raise ConnectionError("dropped")

    fresh = tmp_path / "fresh.csv"
    with pytest.raises(ConnectionError):
        copy_query(_FakeConn(_broken()), "SELECT 1", fresh)
    assert not fresh.exists() and not fresh.with_name(fresh.name + ".part").exists()
