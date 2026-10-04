"""Conditional four-stage progress curriculum, gated by measured angle."""
import math


class ProgressCurriculum:
    targets = (math.radians(10), math.radians(30), math.radians(50), 1.0)

    def __init__(self, threshold=0.8, consecutive=2):
        self.stage = 0
        self.threshold = threshold
        self.consecutive = consecutive
        self.streak = 0
        self.history = []

    @property
    def target(self):
        return self.targets[self.stage]

    def observe(self, measured_success, steps):
        self.streak = self.streak+1 if measured_success >= self.threshold else 0
        if self.streak >= self.consecutive and self.stage < len(self.targets)-1:
            self.history.append(dict(stage=self.stage, target=self.target, exit_steps=steps,
                                     success=measured_success))
            self.stage += 1
            self.streak = 0
            return True
        return False


def stable_d_gate(summary):
    """Prerecorded gate: protection alone cannot establish stable learning."""
    d = summary["D"]
    return (d["raw_final"]["mean"] >= 0.8 and min(d["raw_final"]["per_seed"]) >= 0.6
            and d["raw_gap"]["mean"] <= 0.1
            and min(d["online_successes"]["per_seed"]) >= 5)
