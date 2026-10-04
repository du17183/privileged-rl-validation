"""Initialize installed Isaac import runtime; diagnostics remain offline CPU-only."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(headless=True).app
try:
    from experiments.phase11_parameter_generalization.diagnose import main
    main()
finally:app.close()
