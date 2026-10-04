"""Five frozen-anchor checks in parallel, with no optimizer updates."""
import json
import os
import subprocess
from pathlib import Path
from experiments.phase9_safe_online.protocol import PROTOCOL
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase9_safe_online"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    running = []
    for seed in range(5):
        logpath = ROOT/"logs"/"phase9_safe_online"/f"anchor_seed{seed}.log"
        if logpath.exists():
            logpath.rename(logpath.with_name(logpath.name+f'.previous_{os.getpid()}'))
        log = logpath.open("w")
        proc = subprocess.Popen([str(ROOT/".venv"/"bin"/"python"), "experiments/phase9_safe_online/prepare_anchor.py",
            "--seed", str(seed), "--device", "cuda:0"], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(seed)), stdout=log, stderr=subprocess.STDOUT)
        running.append((seed, proc, log))
    for seed, proc, log in running:
        code = proc.wait()
        log.close()
        resultpath = OUT/f'anchor_seed{seed}.json'
        if code or not resultpath.exists():
            raise RuntimeError(f"Anchor seed {seed} failed; inspect its log")
        result = json.loads(resultpath.read_text())
        if result['anchor_success_episodes'] < 48:
            raise RuntimeError('Anchor did not pass data gate')
        print(f"Anchor seed {seed} complete", flush=True)
    (OUT/"anchor_preparation_completed.json").write_text(json.dumps(dict(seeds=5, optimizer_updates=0)))


if __name__ == "__main__":
    main()
