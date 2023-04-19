from agents_models.intentional_agents.tom_one_agents.dom_one_memoization import *


class DoMTwoMemoization(DoMOneMemoization):
    def __init__(self, path_to_memoization_dir, softmax_temp=0.1):
        self.softmax_temp = softmax_temp
        self.config = get_config()
        self.target_table_name = f'DoM_1_memoization_data_softmax_temp_{self.softmax_temp}_seed_{self.config.seed}.csv'
        self.path_to_dir = path_to_memoization_dir
        self.path_to_table = os.path.join(self.path_to_dir, self.target_table_name)
        super().__init__(path_to_memoization_dir)

    def query_table(self, query_parameters: dict):
        return pd.DataFrame

