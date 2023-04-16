from IPOMCP_solver.Solver.ipomcp_config import *
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


columns_to_groupby = ["trial_number", "sender_threshold"]
q_values_columns_to_groupby = ["trial_number", "sender_threshold", "action"]


class DoMTwoBehaviouralDataLoader:
    def __init__(self, softmax_temp=0.1):
        self.config = get_config()
        self.device = self.config.device
        self.softmax_temp = softmax_temp

    def load_behavioural_data(self, batch_size=500, random_state=42):
        q_values = self.load_results("q_values")
        game_results = self.load_results("simulation_results")
        nested_beliefs = self.load_results("beliefs")
        data, labels, all_data, train_loader, test_loader = self.aggregate_results(q_values, game_results, nested_beliefs)
        return data, labels, all_data, train_loader, test_loader

    @staticmethod
    def load_results(directory_name):
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
    def aggregate_results(raw_q_values: pd.DataFrame, raw_game_results: pd.DataFrame, raw_beliefs: pd.DataFrame):
        # Compute average Q-value per action and trial
        aggregated_q_values = raw_q_values.loc[raw_q_values['agent'] == 'DoM(1)_sender']
        # Compute

        raw_game_results.groupby(columns_to_groupby).mean().reset_index()
        # Remove actions with negative value
        filtered_q_values = q_values.drop(q_values[q_values.q_value < -500].index)
        # Remove terminal action as its value is deterministic
        filtered_q_values = filtered_q_values.drop(q_values[q_values.action < 0].index)
        filtered_q_values = filtered_q_values.assign(p1=np.round(filtered_q_values["p1"], 3),
                                                     p2=np.round(filtered_q_values["p2"], 3),
                                                     p3=np.round(filtered_q_values["p3"], 3))
        filtered_q_values = filtered_q_values.assign(observation=np.round(filtered_q_values["observation"], 3))
        columns_to_remove = ['n_visits', 'problem', 'agent', 'seed']
        data_for_model = filtered_q_values.drop(columns_to_remove, axis=1)
        average_q_values = data_for_model.groupby(columns_to_groupby).mean().reset_index()
        data = average_q_values.drop(['q_value'], axis=1).to_numpy(dtype=np.float32)
        labels = average_q_values['q_value'].to_numpy(dtype=np.float32)
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20,
                                                                            random_state=random_state, shuffle=True)
        all_data_sets = {"train_data": train_data, "test_data": test_data, "train_labels": train_labels,
                         "test_labels": test_labels}
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels)),
            batch_size=batch_size, drop_last=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels)),
            batch_size=batch_size, drop_last=True, shuffle=True)
        return data_for_model.drop(['q_value'], axis=1)[columns_to_groupby], data_for_model['q_value'], all_data_sets, train_loader, \
               test_loader

    def get_opponent_name(self) -> str:
        if self.dom_level == "DoM(1)":
            return "receiver" if self.role_name == "sender" else "receiver"
