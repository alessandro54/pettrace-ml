"""Conservative copy proposals, with immutable, resumable image-signature chunks.

dHash: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
The thresholds and RGB confirmation are PetTrace choices, not identity recognition.
"""

from collections import defaultdict
from contextlib import closing, nullcontext
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import zlib

from PIL import Image, ImageChops, ImageOps, ImageStat, __version__ as pillow_version

from pettrace.data.checkpoints import DriveBackup
from pettrace.data.verification import sha256


def signature(image):
    """The same EXIF, dHash and RGB thumbnail representation used by curation."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.load()
    values = image.convert("L").resize((9, 8)).tobytes()
    bits = (values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8))
    return sum(int(value) << bit for bit, value in enumerate(bits)), image.width / image.height, image.resize((64, 64))


def pixel_error(left, right):
    return sum(ImageStat.Stat(ImageChops.difference(left, right)).mean) / 3


def configuration():
    return {"implementation_sha256": sha256(__file__), "pillow": pillow_version,
            "hash_distance": 2, "aspect_difference": 0.01, "rgb_mae": 2.0}


class _Signatures:
    """Closed SQLite chunks are content addressed; only new chunks need backup."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries, self.pending = {}, {}
        for path in sorted(self.root.glob("*.db")):
            if sha256(path) != path.stem:
                raise ValueError("Image signature cache checksum mismatch")
            with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
                for digest, dhash, ratio, error in db.execute("SELECT digest, dhash, ratio, error FROM signatures"):
                    value = (int(dhash), ratio, error)
                    if digest in self.entries and self.entries[digest][:3] != value:
                        raise ValueError("Conflicting cached image signatures")
                    self.entries[digest] = (*value, path)

    def add(self, digest, dhash, ratio, thumb, error):
        self.pending[digest] = (dhash, ratio, thumb, error)
        self.entries[digest] = (dhash, ratio, error, None)

    def thumb(self, digest):
        path = self.entries[digest][3]
        if path is None:
            return self.pending[digest][2]
        with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
            data, = db.execute("SELECT thumb FROM signatures WHERE digest = ?", (digest,)).fetchone()
        return Image.frombytes("RGB", (64, 64), zlib.decompress(data))

    def flush(self):
        if not self.pending:
            return
        with tempfile.TemporaryDirectory(prefix=".signatures-", dir=self.root) as temporary:
            path = Path(temporary) / "chunk.db"
            with closing(sqlite3.connect(path)) as db:
                db.execute("CREATE TABLE signatures (digest TEXT PRIMARY KEY, dhash TEXT, ratio REAL, thumb BLOB, error TEXT)")
                db.executemany("INSERT INTO signatures VALUES (?, ?, ?, ?, ?)",
                    ((digest, str(dhash), ratio, zlib.compress(thumb.tobytes()) if thumb else None, error)
                     for digest, (dhash, ratio, thumb, error) in sorted(self.pending.items())))
                db.commit()
            destination = self.root / f"{sha256(path)}.db"
            path.replace(destination)
        for digest in self.pending:
            self.entries[digest] = (*self.entries[digest][:3], destination)
        self.pending.clear()


def find_near_copies(by_hash, cache_dir, *, drive_root=None, source_root="/content/datasets/public/e3-body"):
    """Compare all unique recorded hashes, including small and conflicting labels.

    Cache reuse refers to bytes previously verified against the inventory; it does
    not verify that a source file has stayed unchanged. Preparation verifies it again.
    Restore only one original per uncached hash when Drive is supplied. No inference,
    label merges or deletions. The caller must supply a single writer per cache.
    """
    config = configuration()
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir).resolve() / key
    backup = None
    if drive_root is not None:
        drive_root = Path(drive_root).resolve()
        if not drive_root.is_dir():
            raise FileNotFoundError("Mount Drive before comparing source images")
        backup = DriveBackup({"signatures": root}, drive_root / "checkpoints/lookup-near" / key)
        backup.restore()
    cache = _Signatures(root)
    representatives = {digest: min(photos, key=lambda row: row["path"]) for digest, photos in by_hash.items()}
    missing = {digest: row for digest, row in representatives.items() if digest not in cache.entries}
    if backup is not None and missing:
        source_root = Path(source_root).resolve()
        members = set()
        for row in missing.values():
            path = Path(row["path"]).resolve()
            relative = path.relative_to(source_root)
            parts = relative.parts
            flat = len(parts) == 4 and parts[2] == "images"
            lcw = (row["source"] == "lcw" and len(parts) == 5
                   and parts[2] == "CatFullDataset" and parts[3].isdigit())
            if parts[:2] != ("source-cache", row["source"]) or not (flat or lcw):
                raise ValueError("Source photo is outside the expected preparation cache")
            if not path.is_file():
                members.add(f"data/{relative.as_posix()}")
        if members:
            print(f"Copias cercanas: recuperando {len(members):,} originales sin firma; sin ZIP ni YOLO.", flush=True)
            roots = {"data": source_root, "prepared": root / "unused-prepared", "reference": root / "unused-reference"}
            DriveBackup(roots, drive_root / "checkpoints/preparation").restore(files=members)
    # Fail before scanning if originals are absent; never silently claim partial coverage.
    for row in missing.values():
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(f"Original needed for visual comparison: {row['path']}")

    pairs, corrupt, bands, seen = [], {}, defaultdict(set), {}
    reused = len(by_hash) - len(missing)
    print(f"Copias cercanas: {len(by_hash):,} hashes · {reused:,} firmas reutilizadas · {len(missing):,} nuevas.", flush=True)
    with backup if backup is not None else nullcontext():
        try:
            for number, (digest, row) in enumerate(sorted(representatives.items()), 1):
                if digest not in cache.entries:
                    data = Path(row["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise ValueError(f"Source photo changed: {row['path']}")
                    try:
                        with Image.open(io.BytesIO(data)) as image:
                            dhash, ratio, thumb = signature(image)
                        cache.add(digest, dhash, ratio, thumb, None)
                    except (OSError, ValueError, Image.DecompressionBombError) as exc:
                        cache.add(digest, 0, 0.0, None, type(exc).__name__)
                dhash, ratio, error, _ = cache.entries[digest]
                if error:
                    corrupt[digest] = error
                else:
                    # With <=2 differing bits, one of three disjoint bands must match.
                    keys = [(shift, (dhash >> shift) & ((1 << width) - 1))
                            for shift, width in ((0, 22), (22, 21), (43, 21))]
                    possible = set().union(*(bands[k] for k in keys))
                    thumb = None
                    for old in sorted(possible):
                        old_hash, old_ratio = seen[old]
                        distance = (dhash ^ old_hash).bit_count()
                        if distance <= config["hash_distance"] and abs(ratio - old_ratio) < config["aspect_difference"]:
                            if thumb is None:
                                thumb = cache.thumb(digest)
                            error = pixel_error(thumb, cache.thumb(old))
                            if error <= config["rgb_mae"]:
                                pairs.append({"left": old, "right": digest, "hash_distance": distance, "rgb_mae": error})
                    for band in keys:
                        bands[band].add(digest)
                    seen[digest] = (dhash, ratio)
                if len(cache.pending) >= 1000:
                    cache.flush()
                    if backup is not None:
                        backup()
                if number % 10000 == 0 or number == len(by_hash):
                    print(f"Copias cercanas: {number:,}/{len(by_hash):,} revisadas.", flush=True)
        finally:
            cache.flush()
    return pairs, corrupt
