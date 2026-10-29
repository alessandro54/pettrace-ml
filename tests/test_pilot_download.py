import io
import json
import random
import zipfile

import pytest

from pettrace import pilot_download as download


class Response:
    def __init__(self, content, start, end, size, status=206):
        self.raw = io.BytesIO(content)
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{size}"}
        self.url = "https://storage.example/archive.zip?temporary-token"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, payload, status=206):
        self.payload, self.status, self.calls = payload, status, []
        self.headers = {}

    def mount(self, *args):
        pass

    def close(self):
        pass

    def get(self, url, *, headers, auth=None, **kwargs):
        value = headers["Range"].removeprefix("bytes=")
        if value.startswith("-"):
            start, end = max(0, len(self.payload) - int(value[1:])), len(self.payload) - 1
        else:
            start, end = map(int, value.split("-"))
        self.calls.append({"auth": auth, "url": url, "start": start, "end": end})
        response = Response(self.payload[start:end + 1], start, end, len(self.payload), self.status)
        self.last_response = response
        return response


def archive():
    stream = io.BytesIO()
    payload = random.Random(42).randbytes(160000)
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("images/test.png", payload)
        z.writestr("metadata/test.json", json.dumps({"path": "images/test.png", "identities": [{"identity": "dog-1"}]}))
    return stream.getvalue(), payload


def test_remote_zip_reads_ranges_and_never_forwards_origin_auth():
    blob, expected = archive()
    session = Session(blob)
    with download.RangeFile("https://source.example/download", auth=("user", "secret"),
                            block_size=8192, cache_blocks=2, session=session) as remote:
        with zipfile.ZipFile(remote) as z:
            remote.prefetch_entry(z.getinfo("images/test.png"))
            assert z.read("images/test.png") == expected
        assert len(remote.cache) <= 2
    assert session.calls[0]["auth"] == ("user", "secret")
    assert all(call["auth"] is None for call in session.calls[1:])


def test_range_unsupported_does_not_read_full_archive():
    blob, _ = archive()
    session = Session(blob, status=200)
    with pytest.raises(ValueError, match="no full fallback"):
        download.RangeFile("https://source.example/download", session=session)
    assert len(session.calls) == 1 and session.last_response.raw.tell() == 0


def test_range_transfer_budget_is_enforced():
    blob, _ = archive()
    with download.RangeFile("https://source.example/download", max_network_bytes=65536,
                            session=Session(blob)) as remote:
        remote.seek(0)
        with pytest.raises(ValueError, match="budget"):
            remote.read(10)


def test_candidate_groups_use_sidecar_ids_and_lcw_individuals():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for i in range(4):
            z.writestr(f"images/{i}.png", b"fixture")
            z.writestr(f"metadata/{i}.json", json.dumps({"path": f"images/{i}.png", "identities": [{"identity": "same-dog"}]}))
            z.writestr(f"CatFullDataset/9/{i}.jpg", b"fixture")
        z.writestr("metadata/multiple.json", json.dumps({"path": "images/0.png", "identities": [{"identity": "a"}, {"identity": "b"}]}))
    stream.seek(0)
    with zipfile.ZipFile(stream) as z:
        dogs, scanned = download._candidate_groups(z, "dogs-world", metadata_limit=10, photos_per_identity=4)
        cats, _ = download._candidate_groups(z, "lcw", metadata_limit=10, photos_per_identity=4)
    assert scanned == 5 and len(dogs["same-dog"]) == 4 and len(dogs) == 1
    assert len(cats["9"]) == 4


def test_pilot_receipt_reuses_verified_images_and_sidecars(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for i in range(4):
            z.writestr(f"images/{i}.png", b"fixture" + bytes([i]))
            z.writestr(f"metadata/{i}.json", json.dumps({"path": f"images/{i}.png", "identities": [{"identity": "one"}]}))
    blob = stream.getvalue()
    original = download.RangeFile
    monkeypatch.setattr(download, "RangeFile", lambda *args, **kwargs: original(*args, **kwargs, session=Session(blob)))
    folder, receipt = download.acquire_pilot(tmp_path, "dogs-world", identities=1, photos_per_identity=4)
    assert len(receipt["file_sha256"]) == 8 and receipt["selected_identities"] == ["one"]
    assert not receipt["extraction_pending"] and "temporary-token" not in json.dumps(receipt)
    _, reused = download.acquire_pilot(tmp_path, "dogs-world", identities=1, photos_per_identity=4)
    assert reused == receipt
    (folder / "metadata/0.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        download.acquire_pilot(tmp_path, "dogs-world", identities=1, photos_per_identity=4)


def test_unsafe_archive_names_rejected():
    for name in ["../outside.jpg", "/outside.jpg", "folder\\outside.jpg"]:
        with pytest.raises(ValueError, match="Unsafe"):
            download._safe_name(name)
