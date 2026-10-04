import os
import signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        if entry.stat().st_uid!=os.getuid() or Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
        arguments=(entry/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'experiments/phase9_safe_online/finish.py' in arguments:
            print('Stopping waiting finish process only:',entry.name,flush=True)
            os.kill(int(entry.name),signal.SIGTERM)
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
