"""Replace only the owned waiting postprocessing helper, never any training."""
import json
import os
import signal
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/phase10_quality_lwd'
stopped=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
        command=(entry/'cmdline').read_bytes().decode().split('\0')
        if 'experiments.phase10_quality_lwd.finish' in command:
            os.kill(int(entry.name),signal.SIGTERM);stopped.append(int(entry.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
log=(ROOT/'logs/phase10_quality_lwd/finish.log').open('a')
proc=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.phase10_quality_lwd.finish'],
    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'postprocessing_helper.json').write_text(json.dumps(dict(stopped_waiting_helpers=stopped,
    new_pid=proc.pid,training_processes_untouched=True),indent=2))
print('Postprocessing helper',proc.pid,'replaces',stopped)
