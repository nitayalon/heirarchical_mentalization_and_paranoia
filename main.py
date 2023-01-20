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
    parser.add_argument('--agent_threshold', type=float, default='0.5', metavar='N',
                        help='set agent threshold (default: 0.5)')
    parser.add_argument('--subject_alpha', type=float, default='0.5', metavar='N',
                        help='set subject reward mixing probability (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    alpha_seq = [0.1, 0.3, 0.5, 0.7, 0.9]  # parameters to control exploration
    thresholds = [0.2, 0.5, 0.8]  # parameters to control threshold of agent
    for i in alpha_seq:
        for k in thresholds:
            config.args.subject_alpha = i
            config.args.agent_threshold = k
            print(f'Now running alpha of {config.args.subject_alpha}')
            print("\n")
            print(f'and threshold of {config.args.agent_threshold}')
            eat_task_simulator = EAT(20, config.seed, 1.0)
            thresholds_probabilities = np.array([1/3, 1/3, 1/3])
            random_number_generator = npr.default_rng(get_config().seed)
            if config.args.agent_threshold is None:
                agent_threshold = random_number_generator.choice(thresholds, p=thresholds_probabilities)
            else:
                agent_threshold = config.args.agent_threshold
            subject_threshold = random_number_generator.choice(thresholds, p=thresholds_probabilities)
            agent = IntentionalAgentSubIntentionalModel(eat_task_simulator.agent_actions, config.softmax_temperature, agent_threshold)
            subject = ToMZeroSubject(eat_task_simulator.subject_actions, config.softmax_temperature,
                                     np.array([thresholds, thresholds_probabilities]).T,
                                     IntentionalAgentSubIntentionalModel(eat_task_simulator.agent_actions,
                                                                         config.softmax_temperature,
                                                                         agent_threshold), config.seed, config.args.subject_alpha)
            experiment_results, agents_q_values, subject_belief = eat_task_simulator.simulate_task(subject, agent)
            experiment_results.to_csv(config.simulation_results_dir + "/" + f'seed_{config.seed}.csv', index=False)
            agents_q_values.to_csv(config.planning_results_dir + "/" + f'seed_{config.seed}.csv', index=False)
            subject_belief.to_csv(config.beliefs_dir + "/" + f'seed_{config.seed}.csv', index=False)
