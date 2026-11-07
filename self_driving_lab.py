"""
Self-driving laboratory simulator: Q-learning for reaction-yield optimization.
Run: python self_driving_lab.py
Dependencies: numpy, matplotlib
This is an educational synthetic simulator, not a real chemical or medical model.
"""
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
TEMP_VALUES = np.linspace(30, 100, 11)       # degrees Celsius
CONC_VALUES = np.linspace(0.1, 1.0, 11)      # relative concentration
ACTIONS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0),
           (0,1), (1,-1), (1,0), (1,1)]

def simulated_yield(temp, concentration, rng, noisy=True):
    """Synthetic reaction response; optimum is hidden from the learning agent."""
    y = 95 * np.exp(-((temp-74)/17)**2 - ((concentration-0.68)/0.23)**2)
    if noisy:
        y += rng.normal(0, 1.2)
    return float(np.clip(y, 0, 100))

def train():
    rng = np.random.default_rng(SEED)
    q = np.zeros((len(TEMP_VALUES), len(CONC_VALUES), len(ACTIONS)))
    episodes, steps = 700, 35
    alpha, gamma = 0.18, 0.92
    epsilon_start, epsilon_end = 1.0, 0.04
    rewards = []

    for ep in range(episodes):
        ti = int(rng.integers(0, len(TEMP_VALUES)))
        ci = int(rng.integers(0, len(CONC_VALUES)))
        epsilon = epsilon_end + (epsilon_start-epsilon_end)*(1-ep/episodes)
        total_yield = 0

        for _ in range(steps):
            if rng.random() < epsilon:
                action = int(rng.integers(0, len(ACTIONS)))
            else:
                action = int(np.argmax(q[ti, ci]))

            dti, dci = ACTIONS[action]
            next_ti = int(np.clip(ti+dti, 0, len(TEMP_VALUES)-1))
            next_ci = int(np.clip(ci+dci, 0, len(CONC_VALUES)-1))

            y = simulated_yield(TEMP_VALUES[next_ti], CONC_VALUES[next_ci], rng)
            reward = y / 100.0
            q[ti, ci, action] += alpha * (
                reward + gamma*np.max(q[next_ti, next_ci]) - q[ti, ci, action]
            )
            ti, ci = next_ti, next_ci
            total_yield += y

        rewards.append(total_yield/steps)

    return q, rewards

def random_baseline():
    rng = np.random.default_rng(SEED+1)
    results = []
    for _ in range(700):
        ti = int(rng.integers(0, len(TEMP_VALUES)))
        ci = int(rng.integers(0, len(CONC_VALUES)))
        ys = []
        for _ in range(35):
            dti, dci = ACTIONS[int(rng.integers(0, len(ACTIONS)))]
            ti = int(np.clip(ti+dti, 0, len(TEMP_VALUES)-1))
            ci = int(np.clip(ci+dci, 0, len(CONC_VALUES)-1))
            ys.append(simulated_yield(TEMP_VALUES[ti], CONC_VALUES[ci], rng))
        results.append(np.mean(ys))
    return results

def main():
    q, learned = train()
    random_results = random_baseline()
    window = 30
    smooth_learned = np.convolve(learned, np.ones(window)/window, mode="valid")
    smooth_random = np.convolve(random_results, np.ones(window)/window, mode="valid")
    plt.plot(smooth_learned, label="Q-learning")
    plt.plot(smooth_random, label="Random baseline")
    plt.xlabel("Training episode")
    plt.ylabel("Average reaction yield (%)")
    plt.title("Q-learning vs random baseline")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=180)
    plt.show()

    # Greedy policy demonstration from the center of the search space
    ti = ci = 5
    for _ in range(40):
        action = int(np.argmax(q[ti, ci]))
        dti, dci = ACTIONS[action]
        ti = int(np.clip(ti+dti, 0, 10))
        ci = int(np.clip(ci+dci, 0, 10))
    print(f"Recommended temperature: {TEMP_VALUES[ti]:.1f} °C")
    print(f"Recommended concentration: {CONC_VALUES[ci]:.2f}")
    print(f"Predicted simulated yield: {simulated_yield(TEMP_VALUES[ti], CONC_VALUES[ci], np.random.default_rng(1), noisy=False):.2f}%")

if __name__ == "__main__":
    main()
