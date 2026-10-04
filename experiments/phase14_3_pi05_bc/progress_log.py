"""Write readable progress for the existing experiment; no model updates."""
import contextlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

from .progress import main as snapshot

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/phase14_3_pi05_bc'


def main():
    while True:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            snapshot()
        status = json.loads(output.getvalue())
        stamp = datetime.now().astimezone().isoformat(timespec='seconds')
        if status['delivery_complete']:
            stage = 'COMPLETE'
        elif status['failures']:
            stage = 'FAILED'
        elif status['formal_complete']:
            stage = 'DIAGNOSTICS_AND_REPORT'
        elif status['validation_tests'] >= status['validation_target']:
            stage = 'INDEPENDENT_TEST'
        else:
            stage = 'CHECKPOINT_VALIDATION'
        fields = [
            f'[{stamp}] stage={stage}',
            f"pi05_training={status['pi05_complete']}/10",
            f"validation={status['validation_tests']}/{status['validation_target']}",
            f"independent_test={status['formal_test']}/{status['test_target']}",
            f"gt_ablation={status['gt_ablation']}/10",
            f"action_diagnosis={status['diagnostic_seeds']}/5",
            f"active={','.join(status['live']) or 'none'}",
            f"errors={len(status['failures'])}",
        ]
        print(' | '.join(fields), flush=True)
        latest = None
        for subdir, pattern in [('validation', '*/*/*.json'),
                                ('test', '*/*.json'),
                                ('gt_ablation', '*/*.json')]:
            for path in (RESULTS / subdir).glob(pattern):
                modified = path.stat().st_mtime_ns
                if latest is None or modified > latest[0]:
                    latest = (modified, path)
        if latest:
            path = latest[1]
            try:
                result = json.loads(path.read_text())
                print(f"[{stamp}] latest={path.relative_to(RESULTS)}"
                      f" | success={result.get('success')}"
                      f" | episodes={result.get('episodes')}"
                      f" | elapsed_s={result.get('elapsed_s')}", flush=True)
            except (OSError, ValueError) as exc:
                print(f'[{stamp}] latest_result_pending={type(exc).__name__}', flush=True)
        if status['failures']:
            print(json.dumps(status['failures'], ensure_ascii=False), flush=True)
        if stage in ('COMPLETE', 'FAILED'):
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
