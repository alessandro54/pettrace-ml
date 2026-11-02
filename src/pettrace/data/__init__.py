"""Dataset acquisition, preparation and versioned snapshots."""


def load(version, *, cache_dir="datasets/build/pettrace-training", drive_root=None):
    """Return a verified refined dataset and provenance, reusing local or Drive data first."""
    from pettrace.data.versions import load_dataset

    return load_dataset(version, cache_dir=cache_dir, drive_root=drive_root)
