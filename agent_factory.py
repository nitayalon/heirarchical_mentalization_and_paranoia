from agents_models.intentional_agents.tom_zero_agents.tom_zero_subjects import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_agent import *
from agents_models.subintentional_agents.subintentional_agents import *
from agents_models.subintentional_agents.subintentional_subject import *


class AgentFactory:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.device
        # Load environmental parameters
        self.softmax_temp = float(self.config.softmax_temperature)
        self.exploration_bonus = float(self.config.get_from_env("uct_exploration_bonus"))
        self.agent_actions = np.arange(0, 1.05, 0.05)
        self.subject_actions = np.array([True, False])
        self.alpha_seq = [0.1, 0.3, 0.5, 0.7, 0.9]  # parameters to control subject orientation
        self.agent_thresholds = [0.0, 0.2, 0.5, 0.8]  # parameters to control threshold of agent
        self.subject_thresholds = [0.2, 0.5, 0.8]  # parameters to control threshold of agent

    def create_experiment_grid(self):
        agent_dom_level = self.config.get_agent_tom_level("agent")
        subject_dom_level = self.config.get_agent_tom_level("subject")
        agent_parameters = self.agent_thresholds if agent_dom_level == "DoM(-1)" else None
        subject_parameters = self.subject_thresholds if subject_dom_level == "DoM(-1)" else self.alpha_seq
        return {"agent_parameters": agent_parameters, "subject_parameters": subject_parameters}

    @staticmethod
    def _create_prior_distribution(support):
        thresholds_probabilities = np.repeat(1 / len(support), len(support))
        return np.array([support, thresholds_probabilities]).T

    def constructor(self, agent_role: str):
        agent = None
        agent_dom_level = self.config.get_agent_tom_level(agent_role)
        if agent_dom_level == "DoM(-1)":
            agent = self.dom_minus_one_constructor(agent_role)
        if agent_dom_level == "DoM(0)":
            agent = self.dom_zero_constructor(agent_role)
        # if agent_name == "DoM(1)":
        #     agent = self.dom_one_constructor(agent_role)
        return agent

    def dom_minus_one_constructor(self, agent_role):
        if agent_role == "agent":
            agent = IntentionalAgentSubIntentionalModel(self.agent_actions, self.softmax_temp)
        else:
            agent = SubIntentionalSubject(self.subject_actions, self.softmax_temp)
        return agent

    def dom_zero_constructor(self, agent_role):
        if agent_role == "agent":
            opponent_model = self.dom_minus_one_constructor("subject")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.subject_thresholds)
            output_agent = DoMZeroAgent(self.agent_actions, self.config.softmax_temperature,
                                        opponent_theta_hat_distribution, opponent_model, self.config.seed)
        else:
            opponent_model = self.dom_minus_one_constructor("agent")
            opponent_theta_hat_distribution = self._create_prior_distribution(self.agent_thresholds)
            output_agent = DoMZeroSubject(self.subject_actions, self.config.softmax_temperature,
                                          opponent_theta_hat_distribution, opponent_model, self.config.seed, None)
        return output_agent

    # def dom_one_constructor(self, agent_role):
    #     if agent_role == "worker":
    #         opponent_model = self.tom_minus_one_constructor("worker")
    #         opponent_theta_hat_distribution = self.budget
    #         self_theta_hat_distribution = self.labor_costs
    #     else:
    #         opponent_model = self.tom_minus_one_constructor("manager")
    #         opponent_theta_hat_distribution = self.labor_costs
    #         self_theta_hat_distribution = self.budget
    #     use_function_approximation = bool(self.config.get_from_general("use_function_approximation"))
    #     return ToMOneAgent(agent_role, self.behavioural_model, use_function_approximation,
    #                        opponent_model, opponent_theta_hat_distribution, self_theta_hat_distribution)
