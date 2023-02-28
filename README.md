# Welfare and Fairness in Multi-objective Reinforcement Learning (AAMAS 2023)

This repository contains our implementation of experiments in the paper, primarily using NumPy.

Abstract: We study fair multi-objective reinforcement learning in which an agent must learn a policy that simultaneously achieves high reward on multiple dimensions of a vector-valued reward. Motivated by the fair resource allocation literature, we model this as an expected welfare maximization problem, for some non-linear fair welfare function of the vector of long-term cumulative rewards. One canonical example of such a function is the Nash Social Welfare, or geometric mean, the log transform of which is also known as the Proportional Fairness objective. We show that even approximately optimal optimization of the expected Nash Social Welfare is computationally intractable even in the tabular case. Nevertheless, we provide a novel adaptation of Q-learning that combines non-linear scalarized learning updates and non-stationary action selection to learn effective policies for optimizing nonlinear welfare functions. We show that our algorithm is provably convergent, and we demonstrate experimentally that our approach outperforms techniques based on linear scalarization, mixtures of optimal linear scalarizations, or stationary action selection for the Nash Social Welfare Objective.

# Requirements

To install the necessary packages, run

```
pip install -r requirements.txt
```

- To reproduce our results for Welfare Q-learning with NSW, mixture policy, and linear scalarization baselines, run those files with same hyper-parameter as reported in the paper, respectively: nsw_ql.py, mixture_policy.py, linear_scalarization.py
- To reproduce our results in the supplementary material for other welfare functions (egalitarian, p-welfare), select the welfare function of your choice in the file, and run other_welfare.py with same hyper-parameters.
