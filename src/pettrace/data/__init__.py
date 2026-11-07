"""Immutable refined datasets used by PetTrace experiments."""

__all__ = ["dogs_of_the_world", "labeled_cats_in_the_wild", "training"]


def dogs_of_the_world(*, version="v1", cache_dir=None, token=None):
    """Download the public refined Dogs of the World release from Hugging Face."""
    from pettrace.data.refined_sources import load_refined_source

    return load_refined_source("dogs-world", version=version, cache_dir=cache_dir, token=token)


def labeled_cats_in_the_wild(*, version="v1", cache_dir=None, token=None):
    """Download the public refined LCW release from Hugging Face."""
    from pettrace.data.refined_sources import load_refined_source

    return load_refined_source("lcw", version=version, cache_dir=cache_dir, token=token)


def training(*, version="v2", cache_dir=None, token=None):
    """Load the combined public refined corpus ready for PetTrace training."""
    from pettrace.data.refined_sources import load_training

    return load_training(version=version, cache_dir=cache_dir, token=token)
