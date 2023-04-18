from agents_models.intentional_agents.tom_one_agents.dom_one_memoization import *


class DoMTwoMemoization(DoMOneMemoization):
    def __init__(self, path_to_memoization_dir, softmax_temp=0.1):
        self.softmax_temp = softmax_temp
        self.config = get_config()
        self.target_table_name = f'DoM_1_memoization_data_softmax_temp_{self.softmax_temp}_seed_{self.config.seed}.csv'
        self.path_to_dir = path_to_memoization_dir
        self.path_to_table = os.path.join(self.path_to_dir, self.target_table_name)
        super().__init__(path_to_memoization_dir)

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
        return pd.DataFrame

    def update_table(self, q_values: np.array, history: np.array, beliefs: np.array, game_parameters: dict):
        trial_data = np.array([game_parameters['trial'], game_parameters['seed'], game_parameters['threshold']])
        beliefs = np.r_[beliefs, np.round(beliefs, 3)]
        data_to_append = pd.DataFrame(np.c_[q_values, np.tile(trial_data, (q_values.shape[0], 1)),
                                            np.tile(beliefs, (q_values.shape[0], 1)),
                                            np.tile(history, (q_values.shape[0], 1))],
                                      columns=self.data.columns)
        self.new_data = pd.concat([self.new_data, data_to_append])
        self.data = pd.concat([self.data, data_to_append])
