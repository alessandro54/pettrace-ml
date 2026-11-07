"""Non-training package operations must work without optional training dependencies."""

import subprocess
import sys


def test_embedding_evaluation_does_not_import_training_dependencies():
    subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
sys.modules['onnx'] = None
import numpy as np
from pettrace.evaluation.metrics import pair_scores, retrieval

features = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
rows = [{'animal': animal, 'species': 'cat'} for animal in ['a', 'a', 'b', 'b']]
animals = [row['animal'] for row in rows]
species = [row['species'] for row in rows]
assert retrieval(features, animals, species)['global']['mAP'] == 1.0
same, different = pair_scores(features, animals, species)
assert same.tolist() == [1.0, 1.0]
assert different.tolist() == [0.0, 0.0, 0.0, 0.0]
"""], check=True, capture_output=True, text=True)
