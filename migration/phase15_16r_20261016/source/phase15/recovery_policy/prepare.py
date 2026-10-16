"""Reuse exact whole-trajectory splits; never regenerate existing experts."""
import json
import shutil
import numpy as np
from pi05.action_adapter import chunk_indices
from phase15.common import ROOT, DATA, CHECKPOINTS, RESULTS, sha256, save_json, historical
from .state import expand, EXTRA_FEATURES, RecoveryAdapter


def main():
    output = DATA / 'manifest.json'
    if output.exists():
        m = json.loads(output.read_text())
        for name, digest in m['input_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError(f'Existing source changed: {name}')
        return
    DATA.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'datasets/phase14_2/prepared_v1'
    m = json.loads((original / 'manifest.json').read_text())
    sources = {}
    inputs = {}
    for source in ['base', 'recovery']:
        for split in ['train', 'validation', 'test']:
            path = original / f'{source}_{split}.npz'
            inputs[str(path.relative_to(ROOT))] = sha256(path)
            with np.load(path) as d:
                raw = np.concatenate((d['robot'], d['environment']), -1)
                sources[source, split] = dict(raw=expand(raw), action=d['action'].copy(),
                                              index=d['trajectory_index'].copy())
    # First 39 frozen train-S statistics. Extra geometry uses ONLY train S/R,
    # identical for R0/R1/R2 and all seeds; validation/test never fitted.
    pool = np.concatenate([sources[s, 'train']['raw'][:, 39:] for s in ['base', 'recovery']])
    mean = np.r_[m['mean'], pool.mean(0)].astype(np.float32)
    std = np.r_[m['std'], np.maximum(pool.std(0), 1e-3)].astype(np.float32)
    adapter = RecoveryAdapter(mean, std)
    counts = {}
    outputs = {}
    for (source, split), d in sources.items():
        tokens, masks = adapter.encode(d['raw'])
        idx, valid = chunk_indices(d['index'])
        if not np.all(d['index'][idx] == d['index'][:, None]):
            raise RuntimeError('Chunk crosses trajectory boundary')
        path = DATA / f'{source}_{split}.npz'
        if path.exists():
            raise RuntimeError('Partial data directory: inspect before overwriting')
        np.savez(path, raw=d['raw'], tokens=tokens.astype(np.int32), token_mask=masks,
                 actions=d['action'][idx], action_mask=valid, index=d['index'])
        outputs[str(path.relative_to(ROOT))] = sha256(path)
        counts[f'{source}_{split}'] = dict(anchors=len(d['raw']),
            trajectories=len(np.unique(d['index'])), max_tokens=int(masks.sum(-1).max()))
    normal = CHECKPOINTS / 'normal_policy'
    normal.mkdir(parents=True, exist_ok=True)
    copied = {}
    for seed in range(5):
        source = historical('C', seed)
        dest = normal / source.name
        if dest.exists() and sha256(dest) != sha256(source):
            raise RuntimeError('Normal checkpoint conflict')
        if not dest.exists():
            shutil.copy2(source, dest)
        copied[str(dest.relative_to(ROOT))] = dict(sha256=sha256(dest),
            original=str(source.relative_to(ROOT)), frozen=True, seed=seed)
    save_json(RESULTS / 'normal_policy_manifest.json', copied)
    save_json(output, dict(input_sha256=inputs, output_sha256=outputs, counts=counts,
        mean=mean.tolist(), std=std.tolist(), input_dim=47, extra_features=EXTRA_FEATURES,
        original_dataset_manifest=m['dataset_sha256'], chunk_horizon=10, execute_steps=5,
        normalization='original train-S first39; appended geometry fitted on train S/R only',
        target_angle=1., source_splits_unchanged=True, rl_updates=0))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
