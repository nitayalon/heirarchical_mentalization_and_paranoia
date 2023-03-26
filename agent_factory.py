from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.subintentional_agents.subintentional_senders import *
from agents_models.subintentional_agents.subintentional_receiver import *


class AgentFactory:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.device
        # Load environmental parameters
        self.softmax_temp = float(self.config.softmax_temperature)
        self.exploration_bonus = float(self.config.get_from_env("uct_exploration_bonus"))
        self.agent_actions = np.arange(0, 1.05, 0.05)
        self.subject_actions = np.array([True, False])
        self.thresholds_seq = [0.0, 0.1, 0.2, 0.4]  # parameters to control threshold of agent
        self.grid_size = 0
        self.include_subject_threshold = self.config.get_from_env("subintentional_type")

    def create_experiment_grid(self):
        subject_parameters = self.thresholds_seq
        subject_grid_size = len(self.thresholds_seq)
        agent_parameters = self.thresholds_seq
        agent_grid_size = len(self.thresholds_seq)
        self.grid_size = agent_grid_size * subject_grid_size
        return {"agent_parameters": agent_parameters,
                "subject_parameters": subject_parameters}

    @staticmethod
    def _create_prior_distribution(support):
        thresholds_probabilities = np.repeat(1 / len(support), len(support))
        return np.array([support, thresholds_probabilities]).T

    def constructor(self, agent_role: str):
        agent = None
        agent_dom_level = self.config.get_agent_tom_level(agent_role)
        if agent_dom_level == "DoM-1":
            agent = self.dom_minus_one_constructor(agent_role)
        if agent_dom_level == "DoM0":
            agent = self.dom_zero_constructor(agent_role)
        if agent_dom_level == "DoM1":
            agent = self.dom_one_constructor(agent_role)
        return agent

    def dom_minus_one_constructor(self, agent_role):
        if agent_role == "sender":
            if self.config.subintentional_agent_type == "uniform":
                agent = UniformRationalRandomSubIntentionalSender(self.agent_actions, self.softmax_temp)
            else:
                agent = SoftMaxRationalRandomSubIntentionalSender(self.agent_actions, self.softmax_temp)
        else:
            agent = BasicSubject(self.subject_actions, self.softmax_temp)
        return agent

    def dom_zero_constructor(self, agent_role):
        if agent_role == "sender":
            opponent_model = self.dom_minus_one_constructor("receiver")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.thresholds_seq)
            output_agent = DoMZeroSender(self.agent_actions, self.config.softmax_temperature, None,
                                         opponent_theta_hat_distribution, opponent_model, self.config.seed)
        else:
            opponent_model = self.dom_minus_one_constructor("sender")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.thresholds_seq)
            output_agent = DoMZeroReceiver(self.subject_actions, self.config.softmax_temperature, None,
                                           opponent_theta_hat_distribution, opponent_model, self.config.seed)
        return output_agent

    def dom_one_constructor(self, agent_role):
        if agent_role == "sender":
            opponent_model = self.dom_zero_constructor("receiver")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.thresholds_seq)
            # output_agent = DoMOneSender(self.agent_actions, self.config.softmax_temperature, None,
            #                             opponent_theta_hat_distribution, opponent_model, self.config.seed)
            output_agent = None
        else:
            opponent_model = self.dom_zero_constructor("sender")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.thresholds_seq)
            output_agent = DoMZeroReceiver(self.subject_actions, self.config.softmax_temperature, None,
                                           opponent_theta_hat_distribution, opponent_model, self.config.seed)
        return output_agent

