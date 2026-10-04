"""Supplemental natural-deviation check; original collector remains immutable.

Importing the existing collector initializes AppLauncher before Isaac modules.
Only its expert class is replaced, then its recorded live-handoff protocol runs.
"""
from recovery_expert import collect_recovery_data as collection
from recovery_expert.phase14_1.recovery_planner import RecoveryPlanner
collection.RecoveryPlanner=RecoveryPlanner

if __name__=='__main__':
    try:collection.main()
    finally:collection.app.close()
