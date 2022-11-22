from eat_environment import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_subjects import *

if __name__ == "__main__":
    eat_task_simulator = EAT(100, 6431, 1.0)
    thresholds = np.array([0.2, 0.5, 0.8])
    thresholds_probabilities = np.array([1/3, 1/3, 1/3])
    agent = IntentionalAgentSubIntentionalModel(eat_task_simulator.actions, None, 0.5, 0.01)
    subject = ToMZeroSubject(eat_task_simulator.actions, None, 0.5, 0.01, np.array([thresholds, thresholds_probabilities]).T,
                             IntentionalAgentSubIntentionalModel(eat_task_simulator.actions, None, None, 0.01))
    results = eat_task_simulator.simulate_task(subject, agent)
