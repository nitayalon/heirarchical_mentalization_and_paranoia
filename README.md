# Hierarchical-modelling
This project is part of the Rethink Paranoia project.
The External Agency Task (EAT) simulates an agent playing against a subject.
The CLI command for simulating a DoM(0) subject vs DoM(-1) is:

--environment env_2 --seed 851 --softmax_temp 0.1 --agent_tom tom0 --subject_tom tom-1 --agent_threshold 0.2 --subject_alpha=0.5
 
--seed: (int) for reproducibility

--softmax_temp: (float) set the SoftMax temperature

--agent_tom: (str) set the agent's ToM level

--agent_threshold: (float,[0,1]) set the agent's threshold

--subject_alpha: (float[0,1]) set the subject's utility function sensitivity to reward/identification
