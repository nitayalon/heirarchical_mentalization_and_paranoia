from typing import Tuple
from agents_models.abstract_agents import *


class EAT:

    def __init__(self, seed):
        self.config = get_config()
        self.n_trails = int(self.config.get_from_env("n_trials"))
        self.seed = seed
        self.endowment = float(self.config.get_from_env("endowment"))
        self.trail_results = []

    def export_beliefs(self, beliefs: Optional[np.array], agent_name: str,
                       subject_threshold: str, agent_threshold: str):
        beliefs_df = None
        if beliefs is not None:
            beliefs_df = pd.DataFrame(beliefs.T[1:, ], columns=beliefs.T[0, ])
            beliefs_df['agent_name'] = agent_name
            beliefs_df['seed'] = self.seed
            beliefs_df['trial_number'] = np.arange(0, beliefs_df.shape[0], 1)
            beliefs_df = self.add_experiment_data_to_df(beliefs_df, subject_threshold, agent_threshold)
        return beliefs_df

    @staticmethod
    def add_experiment_data_to_df(df: pd.DataFrame, subject_threshold: str, agent_threshold: str) -> pd.DataFrame:
        df['subject_threshold'] = subject_threshold
        df['agent_threshold'] = agent_threshold
        return df

    def simulate_task(self, subject, agent, subject_threshold: str, agent_threshold: str) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seed = self.seed
        q_values_list = []
        offer = Action(None, False)
        response = Action(None, False)
        for trial_number in range(1, self.n_trails+1, 1):
            offer, response, trial_results, q_values = self.trial(trial_number, subject, agent, seed, offer, response)
            self.trail_results.append(trial_results)
            q_values_list.append(q_values)
        experiment_results = pd.DataFrame(self.trail_results, columns=['offer', 'response', 'agent_reward',
                                                                       'subject_reward'])
        experiment_results['trial_number'] = np.arange(1, self.n_trails+1, 1)
        experiment_results['seed'] = self.seed
        experiment_results = self.add_experiment_data_to_df(experiment_results, subject_threshold,
                                                            agent_threshold)
        agents_q_values = pd.concat(q_values_list)
        agents_q_values.columns = ['action', 'q_value', 'agent', 'parameter', 'trial_number']
        agents_q_values['seed'] = self.seed
        agents_q_values = self.add_experiment_data_to_df(agents_q_values, subject_threshold,
                                                         agent_threshold)
        subject_belief = self.export_beliefs(subject.belief.belief_distribution, subject.name,
                                             subject_threshold,  agent_threshold)
        agent_belief = self.export_beliefs(agent.belief.belief_distribution, agent.name,
                                           subject_threshold, agent_threshold)
        return experiment_results, agents_q_values, subject_belief, agent_belief

    @staticmethod
    def trial(trial_number, subject, agent, seed, offer, response):
        offer, agent_q_values = agent.act(seed, offer, response, trial_number)
        response, subject_q_values = subject.act(seed, response, offer, trial_number + 1)
        agent_reward = offer.value * response.value
        subject_reward = (1-offer.value) * response.value
        # agent.update_history(offer, response, agent_reward)
        # subject.update_history(response, offer, subject_reward)
        agent_q_values = pd.DataFrame(agent_q_values)
        agent_q_values['agent'] = agent.name
        agent_q_values['parameter'] = agent.threshold
        agent_q_values['trial'] = trial_number
        subject_q_values = pd.DataFrame(subject_q_values)
        subject_q_values['agent'] = subject.name
        subject_q_values['parameter'] = subject.alpha
        subject_q_values['trial'] = trial_number
        q_values = pd.concat([agent_q_values, subject_q_values])
        return offer, response, np.array([offer.value, response.value, agent_reward, subject_reward]), q_values

