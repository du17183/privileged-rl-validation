# Door critic training diagnosis

Five paired seeds, 500k online transitions each, unchanged Phase 2 Door task and expert data.
Training telemetry is sampled after every 32-environment vector step (four SAC gradient updates).
Policy entropy is a Monte Carlo estimate of the tanh-squashed action entropy on the final training minibatch.
The fixed probe uses the same 256 expert transitions at every Phase 2 checkpoint.

## Online training windows

| Variant | Environment steps | Critic loss | Q mean | Q variance | Twin Q disagreement | Actor loss | Policy entropy | Rolling training success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diag_A | 0–100,000 | 0.123 | 3.413 | 3.044 | 0.038 | -3.442 | -1.374 | 0.000 |
| diag_A | 100,000–200,000 | 0.038 | 5.101 | 4.675 | 0.033 | -5.114 | -7.041 | 0.000 |
| diag_A | 200,000–300,000 | 0.020 | 5.215 | 5.767 | 0.031 | -5.221 | -6.995 | 0.000 |
| diag_A | 300,000–400,000 | 0.019 | 4.993 | 6.372 | 0.031 | -5.001 | -7.001 | 0.003 |
| diag_A | 400,000–500,000 | 0.019 | 5.117 | 7.560 | 0.035 | -5.137 | -6.993 | 0.006 |
| diag_B | 0–100,000 | 0.128 | 2.564 | 3.094 | 0.029 | -2.591 | 0.034 | 0.000 |
| diag_B | 100,000–200,000 | 0.034 | 4.344 | 9.793 | 0.037 | -4.364 | -7.081 | 0.004 |
| diag_B | 200,000–300,000 | 0.073 | 7.088 | 17.149 | 0.051 | -7.153 | -7.026 | 0.026 |
| diag_B | 300,000–400,000 | 0.102 | 9.040 | 25.666 | 0.060 | -9.115 | -7.005 | 0.052 |
| diag_B | 400,000–500,000 | 0.124 | 10.139 | 30.075 | 0.070 | -10.227 | -6.999 | 0.069 |

## Fixed expert-state checkpoint probe

| Variant | Steps | Expert Q | Q variance | Policy−expert Q | Twin disagreement | Expert TD absolute error | Action gradient norm | Gradient toward expert cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | -3.421 | 1.362 | -0.008 | 0.037 | 0.850 | 2.544 | 0.184 |
| A | 100,000 | 6.633 | 3.646 | -0.012 | 0.021 | 0.084 | 0.422 | 0.288 |
| A | 200,000 | 8.263 | 1.985 | 0.002 | 0.027 | 0.102 | 0.702 | -0.094 |
| A | 300,000 | 8.398 | 2.403 | 0.011 | 0.023 | 0.062 | 0.829 | 0.109 |
| A | 400,000 | 8.494 | 3.137 | 0.017 | 0.023 | 0.070 | 0.849 | -0.107 |
| A | 500,000 | 8.748 | 3.472 | 0.019 | 0.026 | 0.054 | 0.916 | -0.029 |
| B | 0 | -3.463 | 1.257 | -0.009 | 0.037 | 0.861 | 2.305 | 0.194 |
| B | 100,000 | 7.064 | 6.124 | -0.001 | 0.016 | 0.114 | 0.776 | -0.305 |
| B | 200,000 | 9.520 | 3.835 | 0.023 | 0.022 | 0.091 | 0.852 | -0.265 |
| B | 300,000 | 31.067 | 524.401 | 1.113 | 0.086 | 1.923 | 2.376 | -0.393 |
| B | 400,000 | 139.570 | 16979.024 | 8.264 | 0.365 | 10.051 | 9.742 | -0.334 |
| B | 500,000 | 687.691 | 573357.745 | 31.844 | 1.063 | 69.035 | 39.802 | -0.358 |

## Seed-level collapse timing (Phase 2 checkpoints)

| Seed | Peak step | Peak success | First ≤10% success after peak | First B Q variance >10× A | B Q variance at peak | B Q variance at 500k | B gradient toward expert at 500k |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50,016 | 0.969 | 75008 | 250016 | 7.34 | 2866665.75 | -0.155 |
| 1 | 275,008 | 0.984 | 300000 | not reached | 3.42 | 17.82 | -0.665 |
| 2 | 100,000 | 0.047 | 125024 | not reached | 5.44 | 0.67 | 0.224 |
| 3 | 175,008 | 0.281 | 200000 | not reached | 4.00 | 0.89 | -0.476 |
| 4 | 400,000 | 1.000 | 450016 | 475008 | 2.14 | 103.60 | -0.717 |

## Interpretation boundary

The fixed probe detects drift or miscalibration on a held-constant expert distribution; it does not measure actual on-policy return.
A more positive predicted policy-versus-expert Q advantage with worse evaluated success is evidence of a ranking mismatch, not proof of its mechanism.
Freezing and switching interventions in the Phase 3 ablation provide stronger intervention evidence.
