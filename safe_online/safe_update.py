"""Progress-gated candidate commit; rejection restores the preceding learner."""
import copy


class SafeUpdate:
    def __init__(self):
        self.rejections = 0
        self.acceptances = 0
        self.rollback_events = 0

    def begin(self, agent):
        self.before = copy.deepcopy(agent.checkpoint())

    def compare(self, incumbent, candidate):
        reasons = []
        for mode in ("deterministic", "policy"):
            old, new = incumbent[mode], candidate[mode]
            for key, tolerance in (("success", .10), ("max_angle", .10),
                                   ("final_angle", .10), ("progress", .10)):
                if new[key] < old[key]-tolerance:
                    reasons.append(f"{mode}:{key}")
            if new["regression_event"] > old["regression_event"]+.10:
                reasons.append(f"{mode}:regression_rate")
        return reasons

    def commit(self, agent, incumbent, candidate):
        reasons = self.compare(incumbent, candidate)
        if reasons:
            agent.load_state(self.before)
            self.rejections += 1
            self.rollback_events += 1
            return False, reasons
        self.acceptances += 1
        return True, []

    def score(self, metrics):
        policy, deterministic = metrics["policy"], metrics["deterministic"]
        return (policy["success"], deterministic["success"], policy["progress"],
                -policy["regression_event"])

