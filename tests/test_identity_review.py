import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pettrace.data.lookup import load_candidates, lookup_duplicates
from pettrace.data.review import quarantine_groups
from pettrace.data.sources import write_review
from pettrace.data.verification import sha256
from test_data_lookup import inventory, read, write
from test_e3_prepare import prepare, setup_pipeline
from test_near_copies import images


def confirm(report, target, numbers=(1,)):
    """Explicit fixture annotation of complete groups, never inferred in production."""
    rows = read(report / 'review.csv')
    cases = quarantine_groups(report)
    groups = []
    for number in numbers:
        labels = cases[number - 1]['animals']
        photos = [r for r in rows if r['animal'] in labels]
        hashes = sorted({r['sha256'] for r in photos})
        groups.append(dict(group=number, decision='same_animal', confirmed_identity_count=1,
                           original_labels=labels, image_sha256=hashes, distinct_image_count=len(hashes),
                           original_file_count=len(photos)))
    decisions = dict(lookup_fingerprint=report.name, reviewer='Fixture reviewer', groups=groups)
    target.write_text(json.dumps(decisions))
    return target


def test_review_merges_only_confirmed_group_preserves_original_and_manual_drop(tmp_path):
    raw, rows = inventory(tmp_path)
    # A separate conflicting group remains pending.
    rows.extend({**rows[2], 'animal': f'lcw:cat:{label}', 'sha256': 'e' * 64,
                 'path': f'/not-mounted/{label}.jpg'} for label in ('x', 'y'))
    write(raw, rows)
    parent = lookup_duplicates(raw, tmp_path / 'reports', minimum_photos=2)
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    decisions = confirm(parent, tmp_path / 'decisions.json')
    result = lookup_duplicates(parent / 'review.csv', parent.parent, minimum_photos=2, identity_review=decisions)
    candidates, excluded = load_candidates(result, raw, minimum_photos=2)
    assert result != parent and before == {p.name: p.read_bytes() for p in parent.iterdir()}
    assert (result / 'review.csv').read_bytes() == raw.read_bytes()
    assert len(candidates) == 3 and {r['animal'] for r in candidates} == {'lcw:cat:a'}
    assert all(r['decision'] == '' for r in candidates)  # Identity confirmation does not approve photos.
    assert excluded[rows[0]['path']] == 'manual_drop'
    assert excluded[rows[2]['path']] == 'exact_duplicate'
    assert excluded[rows[-1]['path']] == 'identity_review_pending'
    assert excluded[rows[5]['path']] == 'insufficient_distinct_photos'
    assert lookup_duplicates(parent / 'review.csv', parent.parent, minimum_photos=2, identity_review=decisions) == result
    with pytest.raises(ValueError, match='original lookup'):
        lookup_duplicates(result / 'review.csv', parent.parent, minimum_photos=2, identity_review=decisions)
    (result / 'identity-review.json').write_text('{}')
    with pytest.raises(ValueError, match='Candidate index changed'):
        load_candidates(result, raw, minimum_photos=2)


@pytest.mark.parametrize('invalid', ['fingerprint', 'labels', 'hashes', 'count', 'decision', 'duplicate', 'species'])
def test_reject_stale_partial_or_ambiguous_confirmations(tmp_path, invalid):
    raw, rows = inventory(tmp_path)
    if invalid == 'species':
        rows[5]['sha256'] = rows[2]['sha256']
        write(raw, rows)
    parent = lookup_duplicates(raw, tmp_path / 'reports')
    decision_file = confirm(parent, tmp_path / 'decisions.json')
    decisions = json.loads(decision_file.read_text())
    group = decisions['groups'][0]
    if invalid == 'fingerprint':
        decisions['lookup_fingerprint'] = 'other'
    elif invalid == 'labels':
        group['original_labels'].pop()
    elif invalid == 'hashes':
        group['image_sha256'].pop()
    elif invalid == 'count':
        group['original_file_count'] += 1
    elif invalid == 'decision':
        group['decision'] = 'similar'
    elif invalid == 'duplicate':
        decisions['groups'].append(group)
    decision_file.write_text(json.dumps(decisions))
    with pytest.raises(ValueError):
        lookup_duplicates(parent / 'review.csv', tmp_path / 'corrected', identity_review=decision_file)
    assert not list((tmp_path / 'corrected').glob('*/lookup.json'))


def test_identity_confirmation_reuses_near_audit_without_clearing_near_quarantine(tmp_path, monkeypatch):
    paths = images(tmp_path)
    paths[-1].write_bytes(b'bad image')
    raw = write_review([dict(path=str(p), source='lcw', species='cat', animal=f'lcw:cat:{i}')
                        for i, p in enumerate(paths)], tmp_path / 'review.csv')
    parent = lookup_duplicates(raw, tmp_path / 'reports', near_cache=tmp_path / 'cache')
    original_audit = json.loads((parent / 'audit.json').read_text())
    decisions = confirm(parent, tmp_path / 'decisions.json')
    for path in paths:
        path.unlink()
    monkeypatch.setattr('pettrace.data.near_copies.find_near_copies', lambda *a, **kw: pytest.fail('No image rescan'))
    result = lookup_duplicates(parent / 'review.csv', parent.parent, identity_review=decisions)
    candidates, exclusions = load_candidates(result, raw)
    assert not candidates
    assert all(exclusions[str(p)] == 'near_copy_review_pending' for p in paths[:3])
    assert exclusions[str(paths[-1])] == 'decode_error'
    audit = json.loads((result / 'audit.json').read_text())
    assert audit['near_copy_pairs'] == original_audit['near_copy_pairs']
    assert audit['unreadable_images'] == original_audit['unreadable_images']
    assert audit['near_copy_identity_labels'] == ['lcw:cat:0']


@pytest.mark.parametrize('protection', ['none', 'excluded', 'held_label', 'held_hash'])
def test_preparation_keeps_alias_provenance_and_protects_entire_held_group(tmp_path, monkeypatch, protection):
    from pettrace.data import preparation, volume
    from pettrace.data.sources import discover_body_ids
    config = setup_pipeline(tmp_path, monkeypatch)
    reference = prepare(config)
    monkeypatch.setattr('pettrace.evaluation.research.PUBLIC_FINGERPRINT', reference.fingerprint)
    held = {r['animal'] for r in reference.rows if r['split'] in {'val', 'test'}}
    labels = sorted({r['animal'] for r in reference.rows if r['species'] == 'cat'} - held)
    first, second = labels[:2]
    if protection == 'held_label':
        second = next(a for a in sorted(held) if a.startswith('lcw:cat:'))
    cats = Path(config.source_roots['lcw']) / 'CatFullDataset'
    first_dir, second_dir = cats / first.rsplit(':', 1)[1], cats / second.rsplit(':', 1)[1]
    shutil.copyfile(second_dir / '0.png', first_dir / '0.png')
    if protection == 'held_hash':
        image = next(r['path'] for r in reference.rows if r['species'] == 'cat' and r['split'] == 'test')
        shutil.copyfile(image, second_dir / '1.png')
    rows, sources = [], []
    for source, root in config.source_roots.items():
        found, _ = discover_body_ids(root, source)
        rows.extend(found)
        sources.append({'ref': volume.CAPACITY[source]['ref'], 'sampled_photos': len(found)})
    # Do not mutate reference data; the prepared reference holds its own immutable image copies.
    monkeypatch.setattr(preparation, 'acquire_candidates', lambda *a, **kw: (rows, {'sources': sources, 'skipped_paths': []}))
    raw = write_review(rows, Path(config.data_root) / 'source-50-seed42/review.csv')
    parent = lookup_duplicates(raw, tmp_path / 'indices')
    decisions = confirm(parent, tmp_path / 'decisions.json')
    corrected = lookup_duplicates(parent / 'review.csv', parent.parent, identity_review=decisions)
    aliases = json.loads((corrected / 'identity-review.json').read_text())['aliases']
    seen = []
    original_screen = preparation.screen_animals
    def screen(items, *args, **kwargs):
        seen.extend(items)
        return original_screen(items, *args, **kwargs)
    monkeypatch.setattr(preparation, 'screen_animals', screen)
    excluded = {second: 'Fixture exclusion'} if protection == 'excluded' else config.excluded_animals
    result = prepare(replace(config, source_percent=50, reference_dataset=str(reference.root),
                             excluded_animals=excluded, candidate_index=str(corrected)))
    train = [r for r in result.rows if r['split'] == 'train']
    assert not held & {r['animal'] for r in train}
    if protection == 'none':
        merged = [r for r in train if r['animal'] == min(first, second)]
        assert len(merged) == 9
        assert max(first, second) not in {r['animal'] for r in train}
    else:
        assert not set(aliases.values()) & {r['animal'] for r in train + seen}
    def signature(items):
        return {(r['animal'], r['sha256'], r['split']) for r in items if r['split'] in {'val', 'test'}}
    assert signature(result.rows) == signature(reference.rows)
    assert (result.root / 'review.csv').read_bytes() == raw.read_bytes()
    assert (result.root / 'identity-review.json').read_bytes() == (corrected / 'identity-review.json').read_bytes()
    assert json.loads((result.root / 'audit.json').read_text())['identity_aliases'] == aliases
    assert json.loads((result.root / 'prepared.json').read_text())['files']['identity-review.json'] == sha256(corrected / 'identity-review.json')
