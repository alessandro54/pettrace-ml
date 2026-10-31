"""Bounded, resumable public pilot downloads using ZIP indexes and HTTP byte ranges.

The full archive is never downloaded. Servers without Range support are rejected.
"""

import hashlib
import io
import itertools
import json
import os
import random
import re
import threading
import time
import zipfile
import zlib
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pettrace.data.sources import SOURCES
from pettrace.data.curation import EXTENSIONS, sha256


def _session():
    session = requests.Session()
    session.headers.update({"User-Agent": "pettrace-e3-research/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
    return session


def _photo_windows(remote, infos, workers):
    """Fetch a bounded window of images concurrently; ZIP decoding stays on the caller thread."""
    if workers == 1:
        for info in infos:
            remote.prefetch_entry(info)
            yield info
        return
    sessions, guard, local = [], threading.Lock(), threading.local()

    def fetch(info):
        if not hasattr(local, "session"):
            local.session = _session()
            with guard:
                sessions.append(local.session)
        # These sessions never receive Kaggle origin credentials.
        return remote.fetch_entry_window(info, session=local.session)

    pending, remaining = deque(), iter(infos)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for info in itertools.islice(remaining, workers):
            pending.append((info, pool.submit(fetch, info)))
        while pending:
            info, future = pending.popleft()
            start, window = future.result()
            remote.window_start, remote.window = start, window
            following = next(remaining, None)
            if following is not None:
                pending.append((following, pool.submit(fetch, following)))
            yield info
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for session in sessions:
            session.close()


def _check_entry(info, maximum=20 << 20):
    _safe_name(info.filename)
    if (info.file_size > maximum or info.compress_size > maximum + (1 << 20)
            or (info.external_attr >> 16) & 0o170000 == 0o120000):
        raise ValueError("Oversized image/metadata or symlink in archive")


class RangeFile(io.RawIOBase):
    """Seekable ZIP input with a small LRU cache; never fall back to a whole GET."""

    def __init__(self, url, *, auth=None, block_size=4 << 20, cache_blocks=4, max_network_bytes=3 << 30, session=None):
        super().__init__()
        self.session = session or _session()
        self.session.headers.update({"User-Agent": "pettrace-e3-research/1.0"})
        self._network_lock = threading.Lock()
        self.block_size, self.cache_blocks = block_size, cache_blocks
        self.position, self.transferred = 0, 0
        self.max_network_bytes = max_network_bytes
        self.window_start, self.window = 0, b""
        self.cache = OrderedDict()
        # requests strips origin credentials on cross-host redirects. Subsequent signed URL
        # requests deliberately receive no Kaggle auth, and signed URLs never enter receipts.
        with self.session.get(url, headers={"Range": "bytes=-65536", "Accept-Encoding": "identity"},
                              auth=auth, stream=True, timeout=(30, 120)) as response:
            if response.status_code != 206:
                raise ValueError(f"Server returned HTTP {response.status_code}; partial ZIP downloads required (no full fallback)")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match:
                raise ValueError("Missing or invalid Content-Range")
            start, end, self.size = map(int, match.groups())
            if end != self.size - 1 or end - start + 1 > 65536:
                raise ValueError("Invalid ZIP tail range")
            if end - start + 1 > self.max_network_bytes:
                raise ValueError("Pilot network budget exhausted before reading ZIP tail")
            self.url = response.url
            self.tail_start = start
            self.tail = response.raw.read(end - start + 2)
            if len(self.tail) != end - start + 1:
                raise ValueError("Truncated ZIP tail")
            self.transferred += len(self.tail)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def _block(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        start = number * self.block_size
        end = min(self.size, start + self.block_size) - 1
        block = self._fetch_range(start, end, self.session)
        self.cache[number] = block
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return block

    def _fetch_range(self, start, end, session):
        size = end - start + 1
        with self._network_lock:
            if self.transferred + size > self.max_network_bytes:
                raise ValueError("Pilot network budget exhausted; reduce pilot size")
            # Reserve before starting I/O so simultaneous workers cannot overspend.
            self.transferred += size
        with session.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                         stream=True, timeout=(30, 120)) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Invalid archive range; no full-download fallback")
            window = response.raw.read(size + 1)
            if len(window) != size:
                raise ValueError("Truncated archive range")
        return window

    def _entry_bounds(self, info):
        _check_entry(info)
        start = info.header_offset
        end = min(self.size - 1, start + 30 + len(info.filename.encode("utf-8")) + 65535 + info.compress_size)
        return start, end

    def fetch_entry_window(self, info, *, session):
        start, end = self._entry_bounds(info)
        return start, self._fetch_range(start, end, session)

    def prefetch_entry(self, info):
        start, end = self._entry_bounds(info)
        if self.window_start <= start and end < self.window_start + len(self.window):
            return
        self.window_start, self.window = self.fetch_entry_window(info, session=self.session)

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        if size > 256 << 20:
            raise ValueError("ZIP index/read exceeds the 256 MiB per-read bound")
        if self.window_start <= self.position and self.position + size <= self.window_start + len(self.window):
            start = self.position - self.window_start
            self.position += size
            return self.window[start:start + size]
        if self.position >= self.tail_start and self.position + size <= self.size:
            start = self.position - self.tail_start
            self.position += size
            return self.tail[start:start + size]
        parts = []
        while size:
            number, offset = divmod(self.position, self.block_size)
            block = self._block(number)
            part = block[offset:offset + size]
            if not part:
                raise ValueError("Empty archive block")
            parts.append(part)
            self.position += len(part)
            size -= len(part)
        return b"".join(parts)

    def close(self):
        if not self.closed:
            self.session.close()
        super().close()


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive entry")
    return path


def _payload(z, info, maximum=20 << 20):
    _check_entry(info, maximum)
    with z.open(info) as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("Archive entry exceeds bound")
    return payload


def _candidate_groups(z, source, *, metadata_limit, photos_per_identity):
    groups = defaultdict(list)
    infos = z.infolist()
    by_name = {info.filename: info for info in infos}
    scanned = 0
    if source == "lcw":
        for info in infos:
            path = _safe_name(info.filename)
            parts = path.parts
            if "CatFullDataset" not in parts or path.suffix.lower() not in EXTENSIONS:
                continue
            index = parts.index("CatFullDataset")
            if len(parts) == index + 3 and parts[index + 1].isdigit():
                groups[parts[index + 1]].append((info, None, None))
    else:
        sidecars = sorted((info for info in infos if PurePosixPath(info.filename).parent.name == "metadata"
                           and info.filename.endswith(".json")), key=lambda info: info.header_offset)
        for info in sidecars[:metadata_limit]:
            scanned += 1
            raw = _payload(z, info, maximum=2 << 20)
            metadata = json.loads(raw)
            identities = metadata.get("identities", [])
            if not isinstance(identities, list) or len(identities) != 1 or not isinstance(identities[0], dict):
                continue
            identity = identities[0].get("identity")
            image_path = metadata.get("path")
            if not identity or not isinstance(image_path, str):
                continue
            base = PurePosixPath(info.filename).parent.parent
            image_name = str(base / _safe_name(image_path))
            image = by_name.get(image_name)
            if image and PurePosixPath(image_name).suffix.lower() in EXTENSIONS:
                # Cap retained metadata in memory; scanning still counts all eligible files.
                if len(groups[str(identity)]) < photos_per_identity:
                    groups[str(identity)].append((image, info, raw))
            if scanned % 10000 == 0:
                print(f"Dog metadata scanned: {scanned}/{min(len(sidecars), metadata_limit)}", flush=True)
    return groups, scanned


def acquire_pilot(root, source, *, identities=300, photos_per_identity=10, minimum_photos=4,
                  seed=42, metadata_limit=100000, max_photo_bytes=2 << 30, max_network_bytes=3 << 30, workers=4):
    """Download a bounded identity pilot; no inference/training or automatic review decisions."""
    if (source not in SOURCES or identities < 1 or minimum_photos < 2 or photos_per_identity < minimum_photos
            or not isinstance(workers, int) or not 1 <= workers <= 16):
        raise ValueError("Invalid pilot request")
    spec = SOURCES[source]
    root = Path(root).resolve()
    directory = root / source
    receipt_path = root / f"{source}-source.json"
    config = {"identities": identities, "photos_per_identity": photos_per_identity,
              "minimum_photos": minimum_photos, "seed": seed, "metadata_limit": metadata_limit,
              "max_photo_bytes": max_photo_bytes, "max_network_bytes": max_network_bytes}
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved.get("mode") != "http-range-pilot" or saved.get("config") != config:
            raise ValueError("Different source/config already exists; use a new pilot data root")
        if not saved.get("extraction_pending"):
            for name, digest in saved["file_sha256"].items():
                path = directory / name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError("Cached pilot image changed; use a new data root")
            return directory, saved
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("Existing extraction without pilot receipt; supply it through SOURCE_ROOTS instead")
    root.mkdir(parents=True, exist_ok=True)
    receipt = {**spec, "mode": "http-range-pilot", "config": config, "extraction_pending": True}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    auth = None
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['ref']}?datasetVersionNumber={spec['version']}"
    with RangeFile(url, auth=auth, max_network_bytes=max_network_bytes) as remote, zipfile.ZipFile(remote) as z:
        signature = hashlib.sha256()
        for info in z.infolist():
            signature.update(json.dumps([info.filename, info.CRC, info.file_size, info.compress_size,
                                         info.header_offset], separators=(",", ":")).encode() + b"\n")
        groups, scanned = _candidate_groups(z, source, metadata_limit=metadata_limit,
                                             photos_per_identity=photos_per_identity)
        eligible = sorted(identity for identity, files in groups.items() if len(files) >= minimum_photos)
        rng = random.Random(f"{seed}:{source}")
        rng.shuffle(eligible)
        if len(eligible) < identities:
            raise ValueError(f"Only {len(eligible)} eligible {source} identities in the scanned index; "
                             "lower the requested count or raise metadata_limit using a new root")
        selected = []
        for identity in eligible[:identities]:
            photos = sorted(groups[identity], key=lambda item: item[0].filename)
            rng.shuffle(photos)
            selected.extend(photos[:photos_per_identity])
        if sum(info.file_size for info, _, _ in selected) > max_photo_bytes:
            raise ValueError("Selected photos exceed download budget; reduce the pilot in a new root")
        hashes, plan = {}, []
        for info, sidecar, raw in sorted(selected, key=lambda item: item[0].header_offset):
            _check_entry(info)
            target = directory / str(_safe_name(info.filename))
            valid_cache = target.is_file() and target.stat().st_size == info.file_size and zlib.crc32(target.read_bytes()) == info.CRC
            plan.append((info, sidecar, raw, target, valid_cache))
        missing = [info for info, _, _, _, valid in plan if not valid]
        cached = len(plan) - len(missing)
        print(f"{source}: {cached}/{len(plan)} cached photos verified; downloading {len(missing)} with {workers} workers", flush=True)
        windows = _photo_windows(remote, missing, workers)
        started = last_update = time.monotonic()
        downloaded = 0
        try:
            for info, sidecar, raw, target, valid_cache in plan:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not valid_cache:
                    fetched = next(windows)
                    assert fetched.filename == info.filename
                    payload = _payload(z, info)
                    partial = target.with_suffix(target.suffix + ".partial")
                    partial.write_bytes(payload)
                    partial.replace(target)
                    downloaded += 1
                hashes[str(target.relative_to(directory))] = sha256(target)
                if sidecar:
                    destination = directory / str(_safe_name(sidecar.filename))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(raw)
                    hashes[str(destination.relative_to(directory))] = hashlib.sha256(raw).hexdigest()
                now = time.monotonic()
                if not valid_cache and downloaded and (now - last_update >= 10 or downloaded == len(missing)):
                    rate = downloaded / max(now - started, .001)
                    eta = (len(missing) - downloaded) / rate
                    print(f"{source}: {cached + downloaded}/{len(plan)} photos verified | "
                          f"{rate:.2f} new photos/s | ETA {eta / 60:.1f} min", flush=True)
                    last_update = now
        finally:
            windows.close()
        receipt.update({"extraction_pending": False, "archive_index_sha256": signature.hexdigest(),
                        "archive_bytes": remote.size, "network_bytes": remote.transferred,
                        "metadata_scanned": scanned, "eligible_identities_in_scan": len(eligible),
                        "selected_identities": sorted(eligible[:identities]), "file_sha256": hashes,
                        "note": "Dog selection samples an archive-order metadata prefix, not the entire source uniformly; "
                                "photo review and duplicate audit remain mandatory."})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return directory, receipt
