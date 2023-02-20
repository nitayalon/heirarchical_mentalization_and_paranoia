import pandas as pd
from agents_models.abstract_agents import *


class EAT:

    def __init__(self, n_trails, seed, endowment):
        self.n_trails = n_trails
        self.seed = seed
        self.endowment = endowment
        self.trail_results = []

    def simulate_task(self, subject, agent):
        seed = self.seed
        offer = 1.1
        response = False
        agent.belief.update_history(offer, response)
        subject.belief.update_history(response, offer)
        q_values_list = []
        for trial_number in range(self.n_trails):
            offer, response, trial_results, q_values = self.trial(trial_number, offer, response, subject, agent, seed)
            self.trail_results.append(trial_results)
            q_values_list.append(q_values)
        experiment_results = pd.DataFrame(self.trail_results, columns=['offer', 'response', 'agent_reward',
                                                                       'subject_reward'])
        experiment_results['trial_number'] = np.arange(0, self.n_trails)
        agents_q_values = pd.concat(q_values_list)
        agents_q_values.columns = ['action', 'q_value', 'agent', 'parameter', 'trial_number']
        subject_belief = pd.DataFrame(subject.belief.belief_distribution.T)
        return experiment_results, agents_q_values, subject_belief

    @staticmethod
    def trial(trial_number, offer, response, subject, agent, seed):
        offer, agent_q_values = agent.act(seed, offer, response, trial_number)
        response, subject_q_values = subject.act(seed, response, offer, trial_number)
        agent_reward = offer * response
        subject_reward = (1-offer) * response
        agent_q_values = pd.DataFrame(agent_q_values)
        agent_q_values['agent'] = agent.name
        agent_q_values['parameter'] = agent.threshold
        agent_q_values['trial'] = trial_number
        subject_q_values = pd.DataFrame(subject_q_values)
        subject_q_values['agent'] = subject.name
        subject_q_values['parameter'] = subject.alpha
        subject_q_values['trial'] = trial_number
        q_values = pd.concat([agent_q_values, subject_q_values])
        return offer, response, np.array([offer, response, agent_reward, subject_reward]), q_values

