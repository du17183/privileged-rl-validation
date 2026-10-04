"""Completed-episode replay with exact transition-level normalization.

Uniform B and weighted/selected variants share identical eligibility and pool.
Sampling first chooses a trajectory proportional to length * quality weight,
then a uniformly random transition. Every transition retains nonzero support.
"""
import numpy as np
import torch
from algorithms.replay import FIELDS
from lwd.simplified_lwd import selection_mask


class QualityReplay:
    def __init__(self, expert, online, expert_records, mode, temperature=1., batch_size=256):
        self.expert, self.online = expert, online
        self.expert_records, self.records = expert_records, []
        self.mode, self.temperature, self.batch_size = mode, float(temperature), batch_size
        self.version, self.cached_version = 0, -1
        self.counts = dict(expert=0, online_success=0, online_failure=0, online_partial=0)
        self.histogram = np.zeros(10, dtype=np.int64)
        self.pending_start = torch.zeros(online.capacity, dtype=torch.bool, device=online.device)
        self.transition_quality = torch.full((online.capacity,), -1., device=online.device)
        self.transition_success = torch.full((online.capacity,), -1, dtype=torch.long, device=online.device)

    def complete(self, record):
        self.records.append(dict(record))
        self.version += 1
        idx = record['start']+torch.arange(record['length'], device=self.online.device)*record['stride']
        self.transition_quality[idx] = record['quality_score']
        self.transition_success[idx] = int(record['success'])

    def _prepare(self):
        if self.cached_version == self.version:
            return
        entries = self.expert_records+self.records
        device = self.online.device
        self.lengths = torch.tensor([r['length'] for r in entries], device=device, dtype=torch.long)
        self.starts = torch.tensor([r['start'] for r in entries], device=device, dtype=torch.long)
        self.strides = torch.tensor([r['stride'] for r in entries], device=device, dtype=torch.long)
        self.scores = torch.tensor([r['quality_score'] for r in entries], device=device)
        self.successes = torch.tensor([r['success'] for r in entries], device=device, dtype=torch.bool)
        uniform = self.lengths.float()/self.lengths.sum()
        if self.mode == 'weighted':
            weights = torch.exp((self.scores-1)/self.temperature)
            targeted = weights*self.lengths
            self.probability = .1*uniform+.9*targeted/targeted.sum()
            self.selected_fraction = 1.
        elif self.mode == 'selected':
            keep = selection_mask(self.scores)
            targeted = keep*self.lengths
            self.probability = .2*uniform+.8*targeted/targeted.sum()
            self.selected_fraction = float(keep.float().mean())
        else:
            self.probability = uniform
            self.selected_fraction = 1.
        self.cdf = self.probability.cumsum(0)
        self.cdf[-1] = 1.
        self.cached_version = self.version

    def _gather(self, ids):
        expert_mask = ids < len(self.expert_records)
        idx = self.starts[ids]+(torch.rand(len(ids), device=ids.device)*self.lengths[ids]).long()*self.strides[ids]
        batch = {k: torch.empty((len(ids), self.online.data[k].shape[1]), device=ids.device) for k in FIELDS}
        for key in FIELDS:
            batch[key][expert_mask] = self.expert.prepared[key][idx[expert_mask]]
            batch[key][~expert_mask] = self.online.data[key][idx[~expert_mask]]
        self.counts['expert'] += int(expert_mask.sum())
        self.counts['online_success'] += int((~expert_mask & self.successes[ids]).sum())
        self.counts['online_failure'] += int((~expert_mask & ~self.successes[ids]).sum())
        bins = (self.scores[ids]*10).long().clamp_max(9)
        self.histogram += torch.bincount(bins, minlength=10).cpu().numpy()
        return batch

    def sample(self):
        self._prepare()
        if self.mode in ('original', 'online'):
            n_expert = self.batch_size//2 if self.mode == 'original' else 0
            n_online = self.batch_size-n_expert
            idx = torch.randint(len(self.online), (n_online,), device=self.online.device)
            chunks = [self.expert.sample(n_expert)] if n_expert else []
            chunks.append({k: v[idx] for k, v in self.online.data.items()})
            self.counts['expert'] += n_expert
            labels = self.transition_success[idx]
            self.counts['online_success'] += int((labels==1).sum())
            self.counts['online_failure'] += int((labels==0).sum())
            self.counts['online_partial'] += int((labels==-1).sum())
            known = self.transition_quality[idx] >= 0
            bins = (self.transition_quality[idx][known]*10).long().clamp_max(9)
            self.histogram += torch.bincount(bins, minlength=10).cpu().numpy()
            return {k: torch.cat([c[k] for c in chunks]) for k in FIELDS}
        ids = torch.searchsorted(self.cdf, torch.rand(self.batch_size, device=self.online.device)).clamp_max(len(self.cdf)-1)
        return self._gather(ids)

    def distribution(self):
        self._prepare()
        p = self.probability
        expert = len(self.expert_records)
        online = self.records
        return dict(draws=dict(self.counts), sampled_quality_histogram=self.histogram.tolist(),
                    histogram_scope='online completed draws only; expert and pending excluded' if self.mode in ('original','online') else 'all completed expert+online draws',
                    trajectories=len(self.scores), completed_online_trajectories=len(online),
                    completed_online_transitions=sum(r['length'] for r in online),
                    eligible_expert_probability=float(p[:expert].sum()),
                    transition_density_ratio_max=float((p/(self.lengths.float()/self.lengths.sum())).max()),
                    trajectory_ess=float(1/p.square().sum()), selected_fraction=self.selected_fraction,
                    quality_mean=float(self.scores.mean()), quality_sd=float(self.scores.std()))
