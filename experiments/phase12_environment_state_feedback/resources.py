"""Admit work only when a GPU has no compute processes outside our job tree.

This is the conservative queue after shared-resource launch was rejected.
No external process is signaled or stopped. Low utilization alone is not idle.
"""
import subprocess
from pathlib import Path
def descendants_of(pid,roots):
 seen=set()
 while pid>1 and pid not in seen:
  if pid in roots:return True
  seen.add(pid)
  try:
   status=(Path('/proc')/str(pid)/'status').read_text()
   pid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
  except (OSError,StopIteration,ValueError):return False
 return False
def idle_readings(own_roots):
 rows=[];uuid_to_index={}
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 for line in raw.splitlines():
  index,uuid,memory,util=[v.strip() for v in line.split(',')]
  uuid_to_index[uuid]=int(index);rows.append((int(index),int(memory),int(util)))
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 blocked=set()
 for line in processes.splitlines():
  uuid,pid=[v.strip() for v in line.split(',')]
  if not descendants_of(int(pid),set(own_roots)):blocked.add(uuid_to_index[uuid])
 return [row for row in rows if row[0] not in blocked],sorted(blocked)
