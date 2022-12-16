import pandas as pd
from agents_models.abstract_agents import *


class EAT:

    def __init__(self, n_trails, seed, endowment):
        self.n_trails = n_trails
        self.seed = seed
        self.endowment = endowment
        self.agent_actions = np.arange(0, 1.05, 0.05)
        self.subject_actions = np.array([True, False])
        self.trail_results = []
        self.subject_posterior_beliefs = []

    def simulate_task(self, subject, agent):
        seed = self.seed
        offer = 1.1
        response = False
        agent.belief.update_history(offer, response)
        subject.belief.update_history(response, offer)
        for trial_number in range(self.n_trails):
            offer, response, trial_results = self.trial(trial_number, offer, response, subject, agent, seed)
            self.trail_results.append(trial_results)
            self.subject_posterior_beliefs.append(subject.belief.belief_distribution)
            seed += 1
            print(f'The updated belief of the subject are {subject.belief.belief_distribution[:,-1]}')
        experiment_results = pd.DataFrame(self.trail_results)
        return experiment_results

    @staticmethod
    def trial(trial_number, offer, response, subject, agent, seed):
        offer = agent.act(seed, offer, response)
        response = subject.act(seed, response, offer, trial_number)
        return offer, response, np.array([offer, response])

