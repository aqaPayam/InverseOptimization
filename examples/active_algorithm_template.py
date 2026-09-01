"""Minimal template for an algorithm evaluated by the active benchmark."""

import numpy as np

from invoptlab.active import ActiveAction, ActiveAlgorithm


class MyActiveAlgorithm(ActiveAlgorithm):
    name = "my-active-algorithm"

    def reset(self, context, rng):
        self.context = context
        self.rng = rng
        self.theta_hat = np.zeros(context.dimension)

    def propose(self, history):
        # Replace this rule with the algorithm's choice of the next s_t.
        query = self.context.query_candidates[len(history) % len(self.context.query_candidates)]
        return ActiveAction(query=query, theta_hat=self.theta_hat)

    def observe(self, observation):
        # Update theta_hat using only public fields:
        # observation.query, observation.observed_decision, and observation.observation_mask.
        pass

    def current_estimate(self):
        return self.theta_hat.copy()


def create_algorithm():
    """Factory path: examples.active_algorithm_template:create_algorithm."""
    return MyActiveAlgorithm()

