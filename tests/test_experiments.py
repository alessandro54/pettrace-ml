"""Non-training package operations must work without optional training dependencies."""

import subprocess
import sys


def test_embedding_evaluation_and_policy_guard_do_not_import_training_dependencies():
    subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
sys.modules['onnx'] = None
import numpy as np
from pettrace.experiments.legacy import evaluate_embeddings, run

features = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
rows = [{'animal': animal, 'species': 'cat'} for animal in ['a', 'a', 'b', 'b']]
result = evaluate_embeddings(features, rows, threshold=0.5)
assert result['retrieval']['global']['mAP'] == 1.0
assert result['verification_at_public_val_eer'] == {'far': 0.0, 'frr': 0.0}
try:
    run('unused.csv', 'unused', 'v4', 'unused', log_mlflow=False)
except ValueError as exc:
    assert 'mandatory' in str(exc)
else:
    raise AssertionError('Publication opt-out was accepted')
"""], check=True, capture_output=True, text=True)
