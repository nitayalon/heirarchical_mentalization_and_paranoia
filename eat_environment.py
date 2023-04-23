from typing import Tuple
from agents_models.abstract_agents import *


class EAT:

    def __init__(self, seed):
        self.config = get_config()
        self.n_trails = int(self.config.get_from_env("n_trials"))
        self.endowment = float(self.config.get_from_env("endowment"))
        self.seed = seed
        self.trail_results = []

    def reset(self):
        self.trail_results = []

    def export_beliefs(self, beliefs: Optional[np.array], support, agent_name: str,
                       receiver_threshold: str, agent_threshold: str):
        beliefs_df = None
        if not pd.isna(beliefs).all():
            beliefs_df = pd.DataFrame(beliefs, columns=support)
            beliefs_df['agent_name'] = agent_name
            beliefs_df['seed'] = self.seed
            beliefs_df['trial_number'] = np.arange(0, beliefs_df.shape[0], 1)
            beliefs_df = self.add_experiment_data_to_df(beliefs_df, receiver_threshold, agent_threshold)
        return beliefs_df

    @staticmethod
    def add_experiment_data_to_df(df: pd.DataFrame, receiver_threshold: str, sender_threshold: str) -> pd.DataFrame:
        df['receiver_threshold'] = receiver_threshold
        df['sender_threshold'] = sender_threshold
        return df

    def simulate_task(self, sender, receiver, receiver_threshold: str, sender_threshold: str) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        seed = self.seed
        q_values_list = []
        offer = Action(None, False)
        response = Action(None, False)
        for trial_number in range(1, self.n_trails+1, 1):
            print(f'Starting trial number {trial_number}', flush=True)
            offer, response, trial_results, q_values = self.trial(trial_number, sender, receiver, seed, offer, response)
            self.trail_results.append(trial_results)
            q_values_list.append(q_values)
            print(f'Ending trial number {trial_number}', flush=True)
        experiment_results = pd.DataFrame(self.trail_results, columns=['offer', 'offer_probability',
                                                                       'response', 'response_probability',
                                                                       'sender_reward', 'receiver_reward'])
        experiment_results['trial_number'] = np.arange(1, self.n_trails+1, 1)
        experiment_results['seed'] = self.seed
        experiment_results = self.add_experiment_data_to_df(experiment_results, receiver_threshold,
                                                            sender_threshold)
        agents_q_values = pd.concat(q_values_list)
        agents_q_values.columns = ['action', 'q_value', 'agent', 'parameter', 'trial_number']
        agents_q_values['seed'] = self.seed
        agents_q_values = self.add_experiment_data_to_df(agents_q_values, receiver_threshold,
                                                         sender_threshold)
        receiver_belief = self.export_beliefs(receiver.belief.belief_distribution,
                                              receiver.belief.support,
                                              receiver.name, receiver_threshold,  sender_threshold)
        sender_belief = self.export_beliefs(sender.belief.belief_distribution,
                                            receiver.belief.support, sender.name, receiver_threshold, sender_threshold)
        return experiment_results, agents_q_values, receiver_belief, sender_belief

    @staticmethod
    def trial(trial_number, sender,  receiver, seed, offer, response):
        offer, offer_probability, sender_q_values, sender_policy = sender.act(seed, offer, response, trial_number)
        response, response_probability , receiver_q_values, receiver_policy = receiver.act(seed, response, offer, trial_number + 1)
        sender_reward = (1-offer.value) * response.value
        receiver_reward = offer.value * response.value
        sender_q_values = pd.DataFrame(sender_q_values)
        sender_q_values['agent_name'] = sender.name
        sender_q_values['parameter'] = sender.threshold
        sender_q_values['trial'] = trial_number
        receiver_q_values = pd.DataFrame(receiver_q_values)
        receiver_q_values['agent_name'] = receiver.name
        receiver_q_values['parameter'] = receiver.threshold
        receiver_q_values['trial'] = trial_number
        q_values = pd.concat([sender_q_values, receiver_q_values])
        return offer, response, np.array([offer.value, offer_probability,
                                          response.value, response_probability,
                                          sender_reward, receiver_reward]), q_values

