"""Report measured end-to-end interval rates separately from configured capacity."""
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def main():
    records = []
    for path in sorted(OUT.glob("eval_P8?_seed?.csv")):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2:
            continue
        rates = []
        for previous, row in zip(rows, rows[1:]):
            seconds = float(row["wall_time_s"])-float(previous["wall_time_s"])
            rates.append(((int(row["env_steps"])-int(previous["env_steps"]))/seconds,
                (int(row["env_steps"])-int(previous["env_steps"])+int(row["cumulative_eval_steps"])-int(previous["cumulative_eval_steps"]))/seconds))
        records.append(dict(run=path.stem[5:], latest_step=int(rows[-1]["env_steps"]),
            completed=(ROOT/"checkpoints"/"phase8_progress_rl"/path.stem[5:]/"completed.json").exists(),
            latest_training_steps_per_s=rates[-1][0], fastest_recorded_training_interval_steps_per_s=max(r[0] for r in rates),
            latest_training_plus_evaluation_steps_per_s=rates[-1][1]))
    active = [r for r in records if not r["completed"]]
    result = dict(configured=dict(gpus=8,max_training_runs_per_gpu=3,max_training_slots=24,
        actual_matrix_runs=20,training_envs_per_run=32,peak_training_envs=640,batch_size=256,
        updates_per_vector_step=4,evaluation_envs_per_run=32,evaluation_rounds=2,evaluation_every_steps=10000),
        active_runs=len(active),latest_interval_training_rate_sum=sum(r["latest_training_steps_per_s"] for r in active),
        latest_interval_training_plus_evaluation_rate_sum=sum(r["latest_training_plus_evaluation_steps_per_s"] for r in active),
        fastest_single_run_recorded_training_rate=max(r["fastest_recorded_training_interval_steps_per_s"] for r in records),
        per_run=records,
        rate_scope="Training rates include SAC updates, checkpoint I/O and evaluation waiting. Summed latest intervals are asynchronous estimates, not synchronized peak or simulator-only FPS.")
    (OUT/"throughput_measurement.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
