from eat_environment import *
import argparse
from IPOMCP_solver.Solver.ipomcp_config import *
from agent_factory import *


def export_beliefs_to_file(table: pd.DataFrame, directory_name, output_directory):
    if table is not None:
        table.to_csv(f'{config.beliefs_dir} + "/{directory_name}/" + {output_directory}', index=False)


def set_experiment_name(subject_alpha, subject_threshold, agent_threshold):
    return f'alpha_{subject_alpha}_subject_gamma_{subject_threshold}_agent_gamma_{agent_threshold}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='basic_task', metavar='N',
                        help='game environment (default: basic_task)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--agent_tom', type=str, default='DoM0', metavar='N',
                        help='set agent tom level (default: DoM0)')
    parser.add_argument('--subject_tom', type=str, default='DoM0', metavar='N',
                        help='set subject tom level (default: DoM0)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--agent_threshold', type=float, default='0.5', metavar='N',
                        help='set agent threshold (default: 0.5)')
    parser.add_argument('--subject_alpha', type=float, default='0.5', metavar='N',
                        help='set subject reward mixing probability (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    factory = AgentFactory()
    agent = factory.constructor("agent")
    subject = factory.constructor("subject")
    experiment_data = factory.create_experiment_grid()
    agent_parameters = experiment_data["agent_parameters"]
    subject_parameters = experiment_data["subject_parameters"]
    for subject_param in subject_parameters:
        for agent_param in agent_parameters:
            # Update individual parameters
            subject.threshold = subject_param[0]
            subject.alpha = subject_param[1]
            agent.threshold = agent_param
            # Initial experiment name
            experiment_name = set_experiment_name(subject.threshold, subject.alpha, agent.threshold)
            config.new_experiment_name(experiment_name)
            print(f'Now running alpha of {subject_param}')
            print("\n")
            print(f'and threshold of {agent_param}')
            eat_task_simulator = EAT(20, config.seed, 1.0)
            experiment_results, agents_q_values, subject_belief, agent_belief = \
                eat_task_simulator.simulate_task(subject, agent)
            experiment_name = config.experiment_name
            output_directory_name = f'experiment_data_{experiment_name}_seed_{config.seed}'
            experiment_results.to_csv(config.simulation_results_dir + "/" + output_directory_name, index=False)
            agents_q_values.to_csv(config.q_values_results_dir + "/" + output_directory_name, index=False)
            export_beliefs_to_file(subject_belief, 'subject_beliefs', output_directory_name)
            export_beliefs_to_file(agent_belief, 'agent_beliefs', output_directory_name)
            print(f'simulation over')
