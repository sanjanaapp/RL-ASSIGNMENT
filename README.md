# RL-ASSIGNMENT
# 🧪 Optimizing Chemical Reaction Yield Using Reinforcement Learning

A Reinforcement Learning-based simulation that optimizes chemical reaction yield by intelligently adjusting reaction temperature and reactant concentration using the Q-Learning algorithm.

## 📌 Project Overview

Chemical reaction yield depends on several experimental parameters, including temperature and reactant concentration. Finding the optimal combination through repeated manual experimentation can be time-consuming.

This project implements a **Self-Driving Laboratory Simulator** using Reinforcement Learning. An intelligent agent learns to identify the optimal experimental conditions by interacting with a simulated chemical environment and maximizing reaction yield.

The project uses **Q-Learning**, a model-free Reinforcement Learning algorithm, to optimize temperature and concentration through repeated interactions.

## 🎯 Objectives

- Develop a simulated chemical reaction environment.
- Apply Q-Learning to optimize reaction conditions.
- Identify temperature and concentration combinations that maximize reaction yield.
- Compare the performance of the trained agent against a random baseline.
- Visualize the learning process using performance curves.

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Reinforcement Learning
- Q-Learning

## 🧠 Reinforcement Learning Formulation

The problem is modeled as a Markov Decision Process (MDP).

| Component | Description |
|---|---|
| Agent | Q-Learning agent |
| Environment | Simulated chemical reaction environment |
| State | Current temperature and concentration |
| Action | Increase, decrease, or maintain temperature and concentration |
| Reward | Reaction yield obtained |
| Goal | Maximize chemical reaction yield |

### State Space

The state is represented by two experimental parameters:

- Temperature
- Reactant concentration

The environment uses an 11 × 11 discretized state space, resulting in 121 possible states.

### Action Space

The agent can perform 9 possible actions by adjusting temperature and concentration:

- Increase temperature
- Decrease temperature
- Maintain temperature
- Increase concentration
- Decrease concentration
- Maintain concentration
- Different combinations of these adjustments

### Reward Function

The reward is based on the reaction yield obtained under the selected experimental conditions.

A higher reaction yield provides a higher reward, encouraging the agent to discover optimal reaction conditions.

## ⚙️ Algorithm: Q-Learning

Q-Learning is a model-free, off-policy Reinforcement Learning algorithm that learns the optimal action-selection policy through interaction with an environment.

The Q-value update equation is:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]

Where:

- Q(s,a): Current Q-value
- α: Learning rate
- γ: Discount factor
- r: Immediate reward
- s': Next state
- a': Next action

An epsilon-greedy strategy is used to balance exploration and exploitation.

## 🔄 Implementation Workflow

1. Initialize the simulated chemical reaction environment.
2. Define the state space and action space.
3. Initialize the Q-table.
4. Select actions using the epsilon-greedy strategy.
5. Perform actions and observe the resulting reaction yield.
6. Calculate the reward.
7. Update the Q-table using the Q-Learning update equation.
8. Repeat training over multiple episodes.
9. Evaluate the trained agent against a random baseline.
10. Visualize the learning performance.

## 📊 Experimental Results

The Q-Learning agent is trained over 700 episodes, with 35 steps per episode.

The performance is compared against a random-action baseline.

| Metric | Q-Learning | Random Baseline |
|---|---:|---:|
| Average final performance | 82.73% | 14.64% |

The simulated environment has a theoretical maximum reaction yield of 95%.

The best observed yield during the experiment was 90.90%.

These results demonstrate that the trained agent learns to identify more effective experimental conditions than random exploration.

*Note: Results are based on a synthetic reaction-yield simulator and do not represent real-world laboratory measurements.*

## 📈 Visualizations

### Learning Curve
<!-- Add your learning curve image below -->
![Learning Curve](images/learning_curve.png)

### Q-Learning vs Random Baseline
<!-- Add your comparison graph below -->
![Performance Comparison](images/performance_comparison.png)

### Optimal Reaction Conditions
<!-- Add your results visualization below -->
![Optimal Reaction Yield](images/optimal_yield.png)

## ⚠️ Limitations

- The project uses a simulated chemical environment rather than real laboratory experiments.
- The reaction model is simplified and does not account for all chemical properties.
- The state and action spaces are discretized.
- Performance may vary depending on random initialization and exploration settings.

## 🔬 Ethical Considerations

- The simulation should not be treated as a substitute for real chemical experimentation.
- Real-world implementation would require laboratory validation and safety precautions.
- Experimental conditions must comply with chemical safety standards.
- Results should be interpreted within the limitations of the simulated environment.

## 🚀 Future Enhancements

- Extend the simulation to include additional reaction parameters.
- Explore Deep Q-Networks (DQN) for more complex state spaces.
- Introduce more realistic chemical reaction models.
- Investigate optimization under different experimental constraints.
- Integrate real experimental datasets for validation.

## 👥 Project Information

**Project:** Optimizing Chemical Reaction Yield Using Reinforcement Learning

**Category:** Reinforcement Learning / Self-Driving Laboratory

**Algorithm:** Q-Learning

**Language:** Python

---


