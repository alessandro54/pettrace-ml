"""Reusable PetTrace ML logic for independently executable experiment notebooks.

Data loading, model inference, training, evaluation and tracking live in this package.
Experiment configuration, explanations, figures and findings live in notebooks/.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pettrace-ml")
except PackageNotFoundError:
    __version__ = "0.2.3+source"
