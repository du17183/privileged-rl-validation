# Panda Door Phase 4: stable privileged online RL

The task, USD asset, Panda configuration, reward, automatic reset and 1,000
expert trajectories are inherited unchanged from Phase 2. Phase 4 writes only
to `results/stable_privileged_rl`, `logs/stable_privileged_rl` and
`checkpoints/stable_privileged_rl`.

Run `bash experiments/stable_privileged_rl/launch.sh` on b300-2 for the ten
variants and seeds 0–4. Each run has 32 training environments, 500,000 online
training interactions, and a fixed 32-episode validation every ~10,000 steps.
Evaluation interactions are counted separately. Each validation uses the same
reset seed for a given training seed. `launch_heldout.sh` evaluates the best
and final checkpoints on a separate 64-episode reset sequence. Then run
`.venv/bin/python experiments/stable_privileged_rl/analyze.py` and
`.venv/bin/python evaluation/verify_phase4.py`.
The separately labeled `launch_followup.sh` runs E100R1 (five additional
seeds) after the initial rollback variant showed recurrent collapse.

Validation runs in a persistent **separate Isaac Lab process** on the same
GPU. It receives temporary Actor weights through atomic file requests and
returns 32 fixed-reset episodes. The training environment is never reset by
validation, so full online episodes complete even though validation occurs
every 10k steps (shorter than the roughly 19.2k-interaction episode horizon).
An earlier in-process evaluator interrupted all episodes; its outputs are
preserved under `*_invalid_eval_reset_v1` and excluded from analysis.

The variants and hyperparameters are fixed in `configs/door_phase4.json`.
E0/E25/E50/E100/E200 change only the duration of 0.1-weighted angle/contact
supervision. E100RB restores the complete SAC model and optimizer state after
two consecutive validation drops greater than 0.25 from a best score >=0.5;
the online replay remains. E100M separates angle and contact heads. Q0/QGT/QV
share an observation encoder between the policy and robot-only Q heads. QGT
uses GT supervision for the first 100k steps, and QV additionally weights
completed online trajectories by bounded value rank when recent Q-vs-return
Spearman correlation is at least 0.2. E100R1 is a separately labeled,
post-hoc follow-up after observing E100RB failures: it rolls back after one
drop >0.125, keeping the same 10k evaluation protocol. In all variants,
evaluation and export
call the actor with robot observation alone.

Q variants persist completed and interrupted online trajectories to HDF5 with
`state`, `action`, `reward`, `next_state`, `return`, `value`, `success`, and
`quality_score`; robot and fixture components are also separate. Validation
cannot interrupt learner episodes. The final incomplete training fragments
are marked separately. Both uniform and weighted replay are supported.

The paired comparisons are E25/E50/E100/E200 vs E0, E100RB/E100R1/E100M vs E100,
QGT vs Q0, and QV vs QGT. With five seeds, exact two-sided sign-flip tests
cannot produce p < 0.05; the report presents effect sizes and bootstrap 95%
intervals without claiming significance from those intervals alone.
