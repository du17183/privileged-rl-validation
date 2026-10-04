# Phase14.2 delivery

Complete new source, data splits, two ordinary expert collections, all model
weights, logs, pressure cohorts, episode results, statistics, plots and report.
No RL was trained. No qualified new Anchor was created.

Read docs/phase14_2_recovery_bc_report.md first. Validation-MSE-selected and
final20k checkpoint results are different estimands and both are retained.
evaluation_initialization_v1 is historical debug evidence, excluded from all
final statistics. Existing Phase13/14.1 raw HDF files remain at original paths;
their hashes and references are in dataset manifests. Prepared arrays reproduce
this phase without duplicating those historical raw HDF files.

Source snapshots of unchanged dependencies are under delivery/reused_source_snapshot
and are not extracted over the old project sources. runtime.json records actual
installed packages and GPU inventory. artifact_sha256.json covers all new files
except itself. The external archive SHA256 is saved beside the archive.

Resource logs include obsolete shutdown-stalled workers; worker_cleanup.json
records verified cleanup of only this task's processes after all gates completed.
