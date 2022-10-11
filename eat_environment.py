import numpy as np
import pandas as pd
from agents_models.agent import *


class EAT:

    def __init__(self, n_trails, seed, endowment):
        self.endowment = endowment
        self.n_trails = n_trails
        self.seed = seed
        self.trail_results = []

    def simulate_task(self, subject: Subject, agent: Agent):
        seed = self.seed
        for trial in range(self.n_trails):
            trial_results = self.trial(subject, agent, seed)
            self.trail_results.append(trial_results)
            seed += 1
            print(f'The updated belief of the subject is {subject.posterior_beliefs}')
        experiment_results = pd.DataFrame(self.trail_results)

    def trial(self, subject, agent, seed):
        offer = agent.act(seed)
        response = subject.act(seed, offer)
        agent_reward = self.endowment * offer * response
        subject_reward = self.endowment * (1-offer) * response
        return np.array([offer, response, agent_reward, subject_reward])

