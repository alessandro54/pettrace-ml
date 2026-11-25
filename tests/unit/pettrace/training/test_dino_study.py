import copy

import pytest

from pettrace.experiments.dinov2 import select_family


def candidates():
    return [dict(mode=mode, seed=seed, validation=dict(mAP=.9, rank1=.9))
            for mode in ('projection', 'lora', 'combined') for seed in (45, 46, 47)]


def test_selection_uses_complete_families_and_frozen_control():
    rows = candidates()
    assert select_family(rows, dict(mAP=.9, rank1=.9))['mode'] == 'frozen'
    assert select_family(rows, dict(mAP=.8, rank1=.95))['mode'] == 'frozen'
    selected = select_family(rows, dict(mAP=.8, rank1=.8))
    assert selected['mode'] == 'combined' and selected['seed'] == 45
    with pytest.raises(ValueError, match='exactly'):
        select_family(rows[:-1], dict(mAP=.8, rank1=.8))
    invalid = copy.deepcopy(rows)
    invalid[0]['validation']['mAP'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        select_family(invalid, dict(mAP=.8, rank1=.8))


def test_publication_retry_reuses_run_without_training(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    import pettrace.experiments.dinov2 as study

    class Client:
        created = 0
        fail = True

        def create_run(self, *args, **kwargs):
            self.created += 1
            return SimpleNamespace(info=SimpleNamespace(run_id='same-run'))

        def log_metric(self, *args):
            pass

        def log_artifact(self, *args):
            if self.fail:
                self.fail = False
                raise OSError('temporary upload failure')

        def set_terminated(self, *args):
            pass

        def get_run(self, *args):
            return SimpleNamespace(info=SimpleNamespace(status='FINISHED'))

        def download_artifacts(self, run, artifact, destination):
            from pathlib import Path
            target = Path(destination)/'design.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path/'design.json', target)
            return str(target)

    client = Client()
    monkeypatch.setattr(study, 'production_context', lambda: (client, SimpleNamespace(experiment_id='3')))
    design = dict(mode='lora', seed=45)
    with pytest.raises(OSError):
        study.publish_training(tmp_path, design, dict(mAP=.8))
    assert study.publish_training(tmp_path, design, dict(mAP=.8)) == 'same-run'
    assert client.created == 1
