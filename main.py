from eat_environment import *
import argparse
from IPOMCP_solver.Solver.ipomcp_config import *
from agent_factory import *


def export_beliefs_to_file(table: pd.DataFrame, directory_name, output_directory):
    outdir = os.path.join(config.beliefs_dir, directory_name)
    if table is not None:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        table.to_csv(os.path.join(outdir, output_directory), index=False)


def set_experiment_name(subject_threshold, agent_threshold):
    return f'subject_gamma_{subject_threshold}_agent_gamma_{agent_threshold}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='basic_task', metavar='N',
                        help='game environment (default: basic_task)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--sender_tom', type=str, default='DoM0', metavar='N',
                        help='set sender tom level (default: DoM0)')
    parser.add_argument('--receiver_tom', type=str, default='DoM0', metavar='N',
                        help='set receiver tom level (default: DoM0)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--sender_threshold', type=float, default='0.5', metavar='N',
                        help='set sender threshold (default: 0.5)')
    parser.add_argument('--receiver_alpha', type=float, default='0.5', metavar='N',
                        help='set receiver reward mixing probability (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    factory = AgentFactory()
    sender = factory.constructor("sender")
    receiver = factory.constructor("receiver")
    experiment_data = factory.create_experiment_grid()
    report_point = factory.grid_size
    agent_parameters = experiment_data["agent_parameters"]
    subject_parameters = experiment_data["subject_parameters"]
    i = 0
    for subject_param in subject_parameters:
        for agent_param in agent_parameters:
            # Update individual parameters
            receiver.threshold = subject_param
            sender.threshold = agent_param
            # Initial experiment name
            experiment_name = set_experiment_name(receiver.threshold, sender.threshold)
            config.new_experiment_name(experiment_name)
            print(f'Sender parameters: gamma = {sender.threshold}', flush=True)
            print(f'Receiver parameters: gamma = {receiver.threshold}', flush=True)
            eat_task_simulator = EAT(config.seed)
            experiment_results, agents_q_values, subject_belief, agent_belief = \
                eat_task_simulator.simulate_task(sender, receiver, receiver.threshold, sender.threshold)
            sender.reset(terminal=True)
            receiver.reset(terminal=True)
            experiment_name = config.experiment_name
            output_directory_name = f'experiment_data_{experiment_name}_seed_{config.seed}.csv'
            experiment_results.to_csv(config.simulation_results_dir + "/" + output_directory_name, index=False)
            agents_q_values.to_csv(config.q_values_results_dir + "/" + output_directory_name, index=False)
            export_beliefs_to_file(subject_belief, 'subject_beliefs', output_directory_name)
            export_beliefs_to_file(agent_belief, 'agent_beliefs', output_directory_name)
            print("#" * 10 + ' simulation over ' + "#" * 10, flush=True)
            i += 1
            print(f'{i / factory.grid_size * 100}% of trials completed', flush=True)