# Phase15.1 execution plan

- Preserve Phase1–15 scientific sources, results and weights. New artifacts live in phase15_1, logs/phase15_1 and the two requested docs.
- Freeze the five validation-selected Phase15 Normal/Recovery pairs. No optimizer, RL, new expert training data or task/reward/reset changes.
- First replay seed0's old natural-severe evaluation with observation-only per-tick diagnosis. Finish diagnosis before executing redesigned recovery.
- Compare A old planner; B safe retreat + pose pregrasp + valid grasp with checked straight motion; C adds collision-aware path search; D adds sustained distance/grasp hysteresis. Safety checks remain in every controller.
- Tune on validation cohorts only. Grade deviations at approximately 5/10/15/20cm, at least 100 episodes per grade, with equal cohorts across controllers; report stage denominators.
- Collect additional evaluation-only natural failures of frozen Normal+Recovery on disjoint reset seeds. Preserve prefixes/contact history; retain unsuccessful cases and prevalence. Artificial perturbations never substitute for natural failures.
- Inference baseline already uses BF16 and persistent workers. Check exact state/token caches, unrolled flow, compile and CUDA Graph compatibility without updating any weights. State changes must invalidate caches; do not reuse a state-dependent KV prefix.
- Trained horizon is 10. Executing 5 or 10 ticks uses supported predictions; horizons15/20/30 require actual inference with longer suffixes and explicit out-of-training-horizon status. Never repeat/pad actions to pretend a longer prediction.
- Compare uniform and phase-dependent scheduling, select only on validation, and independently report batch1 forward/service/control latencies, call counts, memory and missed deadlines. Simulation pausing for inference is not real-time deployment evidence.
- Lock candidates before new formal test cohorts; evaluate five frozen pairs with common seeds. Retain all variants and negative results, across-seed CI and paired tests. No uncorrected overall-benefit or causal decoupling claim from Phase15.
- Enter Phase16 only after fixed>=95%, random/full-loop>=90%, artificial>=85%, natural-severe>50% and acceptable actual scheduling; prioritize natural>=60% and artificial>=90%. LWD/DIVL remain deferred.
