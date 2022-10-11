from eat_environment import *
from agents_models.subintentional_agents.intentional_agent import *
from agents_models.subintentional_agents.random_agent import *
from agents_models.intentional_agents.tom_zero_subjects import *

if __name__ == "__main__":
    eat_task_simulator = EAT(10, 6431, 5.0)
    agent = IntentionalAgent(-2.5, 5.0)
    # agent = RandomAgent(-2.5)
    subject = ToMZeroSubject(0.0, np.array([0.5, 0.5]), 5.0, 0.1)
    eat_task_simulator.simulate_task(subject, agent)
