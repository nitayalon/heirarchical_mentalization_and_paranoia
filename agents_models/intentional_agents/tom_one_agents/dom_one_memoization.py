from IPOMCP_solver.utils.memoization_table import *
import os
import numpy as np
import argparse


belief_columns = ["0.0", "0.1", "0.5", "trial_number", "seed", "sender_threshold"]
history_columns = ["offer", "response", "trial_number", "seed", "sender_threshold"]
q_values_columns = ["action", "q_value", "trial_number", "seed", "sender_threshold"]


class DoMOneMemoization(MemoizationTable):
    def __init__(self, softmax_temp=0.1):
        self.softmax_temp = softmax_temp
        super().__init__()

    def load_behavioural_data(self):
        q_values = self.load_results("q_values")
        game_results = self.load_results("simulation_results")
        nested_beliefs = self.load_results("beliefs")
        data = self.combine_results(q_values, game_results, nested_beliefs)
        return data

    @staticmethod
    def load_results(directory_name):
        """
        Method to load tables from memory
        :param directory_name:
        :return:
        """
        data = []
        path = f'data/agent_subject/basic_task/softmax/DoM0_receiver_DoM1_sender_softmax_temp_0.1/{directory_name}'
        if directory_name == "beliefs":
            path = f'data/agent_subject/basic_task/softmax/DoM0_receiver_DoM1_sender_softmax_temp_0.1/{directory_name}/receiver_beliefs'
        files = os.listdir(path)
        for file in files:
            df = pd.read_csv(f'{path}/{file}')
            data.append(df)
        return pd.concat(data, axis=0, ignore_index=True)

    @staticmethod
    def combine_results(raw_q_values: pd.DataFrame, raw_game_results: pd.DataFrame, raw_beliefs: pd.DataFrame):
        """
        Method to merge Q values, history and beliefs into a single view
        :param raw_q_values:
        :param raw_game_results:
        :param raw_beliefs:
        :return:
        """
        # Filter Q-values
        q_values = raw_q_values.loc[raw_q_values['agent'] == 'DoM(1)_sender'][q_values_columns]
        # Compose game history
        history = raw_game_results.loc[raw_game_results['sender_threshold'] > 0][history_columns]
        # filter nested beliefs
        beliefs = raw_beliefs.loc[raw_beliefs['agent_name'] == 'DoM(0)_receiver'][belief_columns]
        # round beliefs
        beliefs = beliefs.assign(p1=np.round(beliefs["0.0"], 3),
                                 p2=np.round(beliefs["0.1"], 3),
                                 p3=np.round(beliefs["0.5"], 3))
        beliefs['trial_number'] = beliefs['trial_number']+1
        # Join tables to get unified view
        q_values_and_beliefs = pd.merge(q_values, beliefs, on=["trial_number", "seed", "sender_threshold"])
        q_values_and_beliefs_and_history = pd.merge(q_values_and_beliefs, history, on=["trial_number", "seed", "sender_threshold"])
        return q_values_and_beliefs_and_history

    def query_table(self, query_parameters: dict):
        """
        Q values memoization: takes as input game settings and returns mean(q-values) if not empty
        :param query_parameters:
        :return: pd.dataframe
        """
        trial = query_parameters['trial']
        threshold = query_parameters['threshold']
        belief = query_parameters['belief']
        belief = np.round(belief, 3)
        p1 = belief[0]
        p2 = belief[1]
        p3 = belief[2]
        results = self.data.loc[(self.data['trial_number'] == trial) & (self.data['sender_threshold'] == threshold) &
                                (self.data['p1'] == p1) & (self.data['p2'] == p2) & (self.data['p3'] == p3)]
        q_values = results.groupby('action')['q_value'].mean().reset_index()
        return q_values

    def update_table(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='basic_task', metavar='N',
                        help='game environment (default: basic_task)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--sender_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_sender tom level (default: DoM0)')
    parser.add_argument('--receiver_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_receiver tom level (default: DoM0)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--sender_threshold', type=float, default='0.5', metavar='N',
                        help='set rational_sender threshold (default: 0.5)')
    parser.add_argument('--receiver_threshold', type=float, default='0.5', metavar='N',
                        help='set rational_receiver threshold (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    data_loader = DoMOneMemoization()
    data_loader.query_table(1, 0.1, np.repeat(1/3, 3))
