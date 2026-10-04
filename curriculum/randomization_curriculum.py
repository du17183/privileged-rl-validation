"""Identical adaptive rule in every arm; realized exposure is logged."""


class RandomizationCurriculum:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.level = 0 if enabled else 2
        self.streak = 0
        self.history = []

    def observe(self, step, independent_scores):
        previous = self.level
        # Two disjoint episode sets at one checkpoint count as two independent
        # evaluations; both must pass. Reset seeds differ, same executing policy.
        if self.enabled and self.level < 3:
            for score in independent_scores:
                self.streak = self.streak+1 if score >= .8 else 0
            if self.streak >= 2:
                self.level += 1
                self.streak = 0
        self.history.append(dict(step=step, previous=previous, level=self.level, scores=independent_scores))
        return self.level != previous
