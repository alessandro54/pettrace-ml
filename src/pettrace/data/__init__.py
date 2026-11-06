"""Dataset acquisition, preparation and versioned snapshots."""


def load(version, *, cache_dir="datasets/build/pettrace-training", drive_root=None):
    """Return a verified refined dataset and provenance, reusing local or Drive data first."""
    from pettrace.data.versions import load_dataset

    return load_dataset(version, cache_dir=cache_dir, drive_root=drive_root)


def dogs_of_the_world(*, version="v1", cache_dir=None, token=None):
    """Download the public refined Dogs of the World release from Hugging Face."""
    from pettrace.data.refined_sources import load_refined_source

    return load_refined_source("dogs-world", version=version, cache_dir=cache_dir, token=token)


def labeled_cats_in_the_wild(*, version="v1", cache_dir=None, token=None):
    """Download the public refined LCW release from Hugging Face."""
    from pettrace.data.refined_sources import load_refined_source

    return load_refined_source("lcw", version=version, cache_dir=cache_dir, token=token)
