from eat_environment import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_subjects import *
from agents_models.subintentional_agents.subintentional_agents import *
import argparse
from IPOMCP_solver.Solver.ipomcp_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='env_2', metavar='N',
                        help='game environment (default: env_2)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--agent_tom', type=str, default='tom0', metavar='N',
                        help='set agent tom level (default: tom0)')
    parser.add_argument('--subject_tom', type=str, default='tom0', metavar='N',
                        help='set subject tom level (default: tom0)')
    parser.add_argument('--softmax_temp', type=float, default='0.5', metavar='N',
                        help='set softmax temp (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    eat_task_simulator = EAT(100, config.seed, 1.0)
    thresholds = np.array([0.2, 0.5, 0.8])
    thresholds_probabilities = np.array([1/3, 1/3, 1/3])
    random_number_generator = npr.default_rng(get_config().seed)
    agent_threshold = random_number_generator.choice(thresholds, p=thresholds_probabilities)
    subject_threshold = random_number_generator.choice(thresholds, p=thresholds_probabilities)
    agent = IntentionalAgentSubIntentionalModel(eat_task_simulator.agent_actions, config.softmax_temperature, agent_threshold)
    subject = ToMZeroSubject(eat_task_simulator.subject_actions, config.softmax_temperature,
                             np.array([thresholds, thresholds_probabilities]).T,
                             IntentionalAgentSubIntentionalModel(eat_task_simulator.agent_actions,
                                                                 config.softmax_temperature,
                                                                 agent_threshold), config.seed)
    results = eat_task_simulator.simulate_task(subject, agent)
