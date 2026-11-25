"""Paired E4 comparisons; fixed galleries, animal-cluster descriptive bootstrap.

References for paired/cluster resampling:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
https://www.stata.com/manuals/rbootstrap.pdf
No model fitting or selection takes place in this analysis.
"""
from collections import defaultdict
from pathlib import PurePosixPath
import numpy as np


def paired_comparison(baseline, candidate, *, resamples=10000, seed=20261118):
    def keyed(evaluation):
        if evaluation['protocol'] != 'zero-shot':
            raise ValueError('Expected the fixed descriptive zero-shot protocol')
        rows = evaluation['test']['per_query']
        values = {(r['species'], r['animal'], '/'.join(PurePosixPath(r['photo']).parts[-2:])): r
                  for r in rows}
        if len(values) != len(rows):
            raise ValueError('Duplicate query identity')
        return values

    left, right = keyed(baseline), keyed(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError('Queries differ')
    clusters = defaultdict(list)
    changes, n01, n10 = [], 0, 0
    for key in sorted(left):
        a, b = left[key], right[key]
        if a['gallery_animals'] != b['gallery_animals']:
            raise ValueError('Gallery identity counts differ')
        hit_a, hit_b = int(a['rank'] == 1), int(b['rank'] == 1)
        delta = [hit_b - hit_a, b['ap'] - a['ap']]
        clusters[key[:2]].append(delta)
        n01 += hit_b > hit_a
        n10 += hit_b < hit_a
        if hit_a != hit_b:
            changes.append({'species': key[0], 'animal': key[1], 'photo': key[2],
                            'before_rank': a['rank'], 'after_rank': b['rank'],
                            'change': 'rescue' if hit_b else 'damage'})
    # Retain every query of a sampled animal; resample animals within species.
    groups = defaultdict(list)
    for (species, animal), values in sorted(clusters.items()):
        groups[species].append([len(values), *np.sum(values, axis=0)])
    rng = np.random.default_rng(seed)
    totals = np.zeros((resamples, 3))
    for species in sorted(groups):
        values = np.asarray(groups[species], dtype=float)
        picks = rng.integers(len(values), size=(resamples, len(values)))
        totals += values[picks].sum(axis=1)
    samples = totals[:, 1:] / totals[:, :1]
    delta = np.sum([np.sum(v, axis=0) for v in clusters.values()], axis=0) / len(left)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {'queries': len(left), 'identities': len(clusters), 'rescues': int(n01),
            'damages': int(n10), 'net_hits': int(n01-n10),
            'delta_rank1': float(delta[0]), 'delta_mAP': float(delta[1]),
            'descriptive_cluster_interval_95': {m: bounds[:, i].tolist()
                                                for i,m in enumerate(('rank1','mAP'))},
            'resamples': resamples, 'bootstrap_seed': seed, 'changed_queries': changes,
            'scope': 'Paired animal clusters, stratified by species; fixed gallery and trained checkpoint. '
                     'Does not quantify gallery resampling, optimization variance, or cross-session generalization.'}


if __name__ == '__main__':
    import copy
    rows = [{'species':'cat','animal':'a','photo':f'a/{i}.jpg','rank':r,'ap':a,'gallery_animals':2}
            for i,r,a in [(0,1,1.0),(1,2,0.5)]]
    baseline = {'protocol':'zero-shot','test':{'per_query':rows}}
    same = paired_comparison(baseline, baseline, resamples=100)
    assert same['net_hits'] == 0 and same['delta_mAP'] == 0
    assert same['descriptive_cluster_interval_95']['rank1'] == [0,0]
    candidate = copy.deepcopy(baseline)
    candidate['test']['per_query'][1].update(rank=1,ap=1.0)
    result = paired_comparison(baseline,candidate,resamples=100)
    assert (result['rescues'],result['damages'],result['net_hits']) == (1,0,1)
    assert result['delta_rank1'] == 0.5 and result['delta_mAP'] == 0.25
    assert result['descriptive_cluster_interval_95']['rank1'] == [0.5,0.5]
    reverse = paired_comparison(candidate,baseline,resamples=100)
    assert reverse['damages'] == 1 and reverse['delta_mAP'] == -0.25
    candidate['test']['per_query'][0]['gallery_animals'] = 3
    try:
        paired_comparison(baseline,candidate,resamples=10)
    except ValueError as exc:
        assert 'Gallery' in str(exc)
    else:
        raise AssertionError('Mismatched galleries accepted')
    print('Paired analysis self-checks passed')
