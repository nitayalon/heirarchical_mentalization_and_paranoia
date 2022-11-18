from eat_environment import *
from agents_models.subintentional_agents.intentional_agent import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_subjects import *

if __name__ == "__main__":
    eat_task_simulator = EAT(100, 6431, 1.0)
    agent = IntentionalAgent(0.5, 1.0)
    # agent = RandomSubIntentionalAgent(4.5, 1.0)
    subject = ToMZeroSubject(0.2, np.array([0.5, 0.5]), 1.0, 0.1)
    results = eat_task_simulator.simulate_task(subject, agent)
