"""Prepare evaluation-only Oxford photos; never trains or evaluates a model."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/build/oxford-distractors-20261120"
CACHE = ROOT / "reports/c1-dinov2-reference/cache"
PRIVATE = CACHE / "private/datasets--alessandro54--pettrace-reid/snapshots/52bfe26384c4230008fe4707aa0cb145988766a8"
SOURCE = ROOT / "datasets/raw/oxford-iiit-pet/mirror/data/train-00000-of-00001.parquet"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pixels(data):
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        digest = sha(str(im.size).encode() + im.tobytes())
        tiny = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        dh = int.from_bytes(np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes(), "big")
    return digest, dh


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Prepared release exists; preserve it")
    receipt = json.loads((CACHE / "dataset-selection.json").read_text())
    paths = {p.stem: p for p in (CACHE / "images").glob("*/*") if p.is_file()}
    benchmark = []
    for r in receipt["rows"]:
        p = paths[r["sha256"]]
        assert sha(p.read_bytes()) == r["sha256"]
        benchmark.append(
            dict(
                domain="public",
                path=str(p.relative_to(ROOT)),
                animal=r["animal"],
                species=r["species"],
                sha256=r["sha256"],
            )
        )
    for r in csv.DictReader((PRIVATE / "metadata.csv").open()):
        p = PRIVATE / r["file_name"]
        assert sha(p.read_bytes()) == r["sha256"]
        benchmark.append(
            dict(
                domain="private",
                path=str(p.relative_to(ROOT)),
                animal=r["animal_id"],
                species=r["species"],
                sha256=r["sha256"],
            )
        )
    assert len(benchmark) == 779
    known = {r["sha256"] for r in benchmark}
    metadata = []
    for repo, commit in [
        ("labeled-cats-in-the-wild-refined", "581e44a04fd461ef1ba248177b84aff61a2ada21"),
        ("dogs-of-the-world-refined", "9fdd6e5f7eaa752f16bd4dae54cd32b88c7e09ba"),
    ]:
        p = (
            Path.home()
            / f".cache/huggingface/hub/datasets--alessandro54--{repo}/snapshots/{commit}/versions/v1/metadata.csv"
        )
        records = list(csv.DictReader(p.open()))
        known.update(r["sha256"] for r in records)
        metadata.append(dict(repo=repo, commit=commit, sha256=sha(p.read_bytes()), rows=len(records)))
    known_pixels, known_dhash = set(), []
    for r in benchmark:
        ph, dh = pixels((ROOT / r["path"]).read_bytes())
        known_pixels.add(ph)
        known_dhash.append(dh)
    rows = pq.read_table(SOURCE).to_pylist()
    for i, r in enumerate(rows):
        r["row"] = i
        r["sha256"] = sha(r["image"]["bytes"])
        r["order"] = sha(("pettrace-oxford-42:" + r["sha256"]).encode())
    selected, excluded = [], []
    for sp, label in [("cat", 0), ("dog", 1)]:
        candidates = sorted((r for r in rows if r["label_cat_dog"] == label), key=lambda r: r["order"])
        retained = 0
        for r in candidates:
            if retained == 500:
                break
            data = r["image"]["bytes"]
            reason = None
            if r["sha256"] in known:
                reason = "byte_duplicate"
            try:
                ph, dh = pixels(data)
            except Exception:
                reason = "decode_failure"
            if not reason and ph in known_pixels:
                reason = "pixel_duplicate"
            if not reason and min((dh ^ x).bit_count() for x in known_dhash) <= 4:
                reason = "dhash_near_duplicate"
            if reason:
                excluded.append(dict(image_id=r["image_id"], row=r["row"], reason=reason, sha256=r["sha256"]))
                continue
            p = OUT / "images" / (r["sha256"] + ".jpg")
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(data)
            selected.append(
                dict(
                    domain="oxford",
                    path=str(p.relative_to(ROOT)),
                    animal="oxford:" + r["image_id"],
                    species=sp,
                    sha256=r["sha256"],
                    image_id=r["image_id"],
                    source_row=r["row"],
                    pixel_sha256=ph,
                    dhash=f"{dh:016x}",
                )
            )
            known.add(r["sha256"])
            known_pixels.add(ph)
            known_dhash.append(dh)
            retained += 1
        assert retained == 500
    manifest = dict(
        source="timm/oxford-iiit-pet",
        commit="089695c834a7deb60505b7cc506672db1c31a6aa",
        split="train",
        source_sha256=sha(SOURCE.read_bytes()),
        metadata_checks=metadata,
        benchmark=benchmark,
        distractors=selected,
        exclusions=excluded,
        identity_disjointness="not certified; byte/pixel/dhash screens only",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                benchmark=len(benchmark),
                distractors=len(selected),
                excluded=len(excluded),
                compared_byte_hashes=len(known),
                manifest_sha256=sha((OUT / "manifest.json").read_bytes()),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
