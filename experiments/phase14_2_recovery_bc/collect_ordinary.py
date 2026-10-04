"""Normal reset trajectories, labeled by the exact Phase14.1 expert.

The historical ordinary collector initializes Isaac before module imports.
Only the planner class is substituted; no task/reward/controller changes.
"""
from recovery_expert import collect_ordinary_data as collection
from recovery_expert.phase14_1.recovery_planner import RecoveryPlanner
collection.RecoveryPlanner=RecoveryPlanner

if __name__=='__main__':
    try:collection.main()
    finally:collection.app.close()
