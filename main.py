from eat_environment import *
from agent_factory import *
import argparse


def export_beliefs_to_file(table: pd.DataFrame, directory_name, output_directory):
    outdir = os.path.join(config.beliefs_dir, directory_name)
    if table is not None:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        table.to_csv(os.path.join(outdir, output_directory), index=False)


def set_experiment_name(receiver_threshold, sender_threshold):
    return f'receiver_gamma_{receiver_threshold}_sender_gamma_{sender_threshold}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--environment', type=str, default='first_task', metavar='N',
                        help='game environment (default: first_task)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--sender_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_sender tom level (default: DoM0)')
    parser.add_argument('--receiver_tom', type=str, default='DoM0', metavar='N',
                        help='set rational_receiver tom level (default: DoM0)')
    parser.add_argument('--duration', type=int, default='10', metavar='N',
                        help='set IUG number of iteration (default: 10)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--sender_threshold', type=float, default='0.5', metavar='N',
                        help='set rational_sender threshold (default: 0.5)')
    parser.add_argument('--receiver_threshold', type=float, default='0.5', metavar='N',
                        help='set rational_receiver threshold (default: 0.5)')
    args = parser.parse_args()
    config = init_config(args.environment, args)
    factory = AgentFactory()
    rational_sender = factory.constructor("rational_sender")
    random_sender = factory.constructor("rational_sender", "DoM-1")
    rational_receiver = factory.constructor("rational_receiver")
    experiment_data = factory.create_experiment_grid()
    report_point = factory.grid_size
    sender_parameters = experiment_data["sender_parameters"]
    i = 0
    eat_task_simulator = EAT(config.seed)
    for sender_threshold in sender_parameters:
        # set random senders
        eat_task_simulator.reset()
        if sender_threshold == 0:
            if config.get_from_general("skip_random"):
                continue
            sender = random_sender            
        else:
            if config.get_from_general("skip_rational"):
                continue
            sender = rational_sender
        receiver = rational_receiver
        # Update individual parameters
        sender.threshold = sender_threshold
        # Initial experiment name
        experiment_name = set_experiment_name(receiver.threshold, sender.threshold)
        config.new_experiment_name(experiment_name)
        print(f'Sender parameters: gamma = {sender.threshold}', flush=True)
        print(f'Receiver parameters: gamma = {receiver.threshold}', flush=True)
        experiment_results, q_values, receiver_belief, sender_belief, receiver_mental_state = \
            eat_task_simulator.simulate_task(sender, receiver, receiver.threshold, sender.threshold)
        if sender.name == "DoM(1)_sender":
            sender.memoization_table.save_data()
        sender.reset()
        receiver.reset()
        experiment_name = config.experiment_name
        output_file_name = f'experiment_data_{experiment_name}_seed_{config.seed}.csv'
        experiment_results.to_csv(config.simulation_results_dir + "/" + output_file_name, index=False)
        q_values.to_csv(config.q_values_results_dir + "/" + output_file_name, index=False)
        export_beliefs_to_file(receiver_belief, 'receiver_beliefs', output_file_name)
        export_beliefs_to_file(sender_belief, 'sender_beliefs', output_file_name)
        export_beliefs_to_file(receiver_mental_state, 'receiver_mental_state', output_file_name)
        print("#" * 10 + ' simulation over ' + "#" * 10, flush=True)
        i += 1
        print(f'{i / factory.grid_size * 100}% of trials completed', flush=True)