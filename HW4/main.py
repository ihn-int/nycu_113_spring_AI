import random
import matplotlib.pyplot as plt
from BanditEnv import BanditEnv
from Agent import Agent

############################################################
# Part 3
# Environment parameters
num_k = 10
num_step = 1000
num_expr = 2000
epsilons = [0, 0.1, 0.01]

# Store experiment datas
average_rewards = {e: [0.0] * num_step for e in epsilons}
optimal_actions = {e: [0.0] * num_step for e in epsilons}

# For each epsilon
for epsilon in epsilons:
    # Create environment
    env = BanditEnv(num_k)
    agent = Agent(num_k, epsilon)

    # For each independent experiment
    for expr in range(num_expr):
        # Reset environment
        env.reset()
        agent.reset()

        # For each step
        for step in range(num_step):
            action = agent.select_action()

            # For optimal arm
            optimal = 1 if action == env.get_optimal_action() else 0

            reward = env.step(action)
            agent.update_q(action, reward)


            # Accumulate rewards and percentages
            average_rewards[epsilon][step] += reward
            optimal_actions[epsilon][step] += optimal

# Calculate the average valuea
for epsilon in epsilons:
    for step in range(num_step):
        average_rewards[epsilon][step] /= num_expr
        optimal_actions[epsilon][step] = (optimal_actions[epsilon][step] / num_expr) * 100

# Plot: reward
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(average_rewards[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_reward_3.png")


# Plot: optimal action
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(optimal_actions[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Optimal Action (%)")
plt.title("Optimal Action Selection over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("opt_action_3.png")

############################################################
# Part 5
# Environment parameters
num_k = 10
num_step = 10000
num_expr = 2000
epsilons = [0, 0.1, 0.01]

# Store experiment datas
average_rewards = {e: [0.0] * num_step for e in epsilons}
optimal_actions = {e: [0.0] * num_step for e in epsilons}

# For each epsilon
for epsilon in epsilons:
    # Create environment
    env = BanditEnv(num_k, stationary=False)
    agent = Agent(num_k, epsilon)

    # For each independent experiment
    for expr in range(num_expr):
        # Reset environment
        env.reset()
        agent.reset()

        # For each step
        for step in range(num_step):
            action = agent.select_action()

            # For optimal arm
            optimal = 1 if action == env.get_optimal_action() else 0

            reward = env.step(action)
            agent.update_q(action, reward)


            # Accumulate rewards and percentages
            average_rewards[epsilon][step] += reward
            optimal_actions[epsilon][step] += optimal

# Calculate the average valuea
for epsilon in epsilons:
    for step in range(num_step):
        average_rewards[epsilon][step] /= num_expr
        optimal_actions[epsilon][step] = (optimal_actions[epsilon][step] / num_expr) * 100

# Plot: reward
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(average_rewards[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_reward_5.png")


# Plot: optimal action
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(optimal_actions[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Optimal Action (%)")
plt.title("Optimal Action Selection over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("opt_action_5.png")

############################################################
# Part 7
# Environment parameters
num_k = 10
num_step = 10000
num_expr = 2000
epsilons = [0, 0.1, 0.01]

# Store experiment datas
average_rewards = {e: [0.0] * num_step for e in epsilons}
optimal_actions = {e: [0.0] * num_step for e in epsilons}

# For each epsilon
for epsilon in epsilons:

    # Create environment
    env = BanditEnv(num_k, stationary=False)
    agent = Agent(num_k, epsilon, alpha=0.1)

    # For each independent experiment
    for expr in range(num_expr):
        # Reset environment
        env.reset()
        agent.reset()

        # For each step
        for step in range(num_step):
            action = agent.select_action()

            # For optimal arm
            optimal = 1 if action == env.get_optimal_action() else 0

            reward = env.step(action)
            agent.update_q(action, reward)


            # Accumulate rewards and percentages
            average_rewards[epsilon][step] += reward
            optimal_actions[epsilon][step] += optimal

# Calculate the average valuea
for epsilon in epsilons:
    for step in range(num_step):
        average_rewards[epsilon][step] /= num_expr
        optimal_actions[epsilon][step] = (optimal_actions[epsilon][step] / num_expr) * 100

# Plot: reward
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(average_rewards[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_reward_7.png")


# Plot: optimal action
plt.figure(figsize=(10, 5))
for epsilon in epsilons:
    plt.plot(optimal_actions[epsilon], label=f"ε = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Optimal Action (%)")
plt.title("Optimal Action Selection over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("opt_action_7.png")