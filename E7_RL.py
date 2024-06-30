#%%
# Import libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

# Create Taxi environment
env = gym.make("Taxi-v3")
 
# Q-table represents the rewards (Q-values) the agent can expect performing a certain action in a certain state
state_space = env.observation_space.n # total number of states
action_space = env.action_space.n # total number of actions
qtable = np.zeros((state_space, action_space)) # initialize Q-table with zeros

# Variables for training/testing
test_episodes = 10000   # number of episodes for testing
train_episodes = 40000  # number of episodes for training
episodes = train_episodes + test_episodes   # total number of episodes
max_steps = 100     # maximum number of steps per episode

# Q-learning algorithm hyperparameters to tune
# Q-learning algorithm hyperparameters to tune
alpha = 1  # learning rate: you may change it to see the difference
gamma = 0.75  # discount factor: you may change it to see the difference


# Exploration-exploitation trade-off
epsilon = 1.0           # probability the agent will explore (initial value is 1.0)
epsilon_min = 0.001     # minimum value of epsilon 
epsilon_decay = 0.9999 # decay multiplied with epsilon after each episode


# TODO:
# Implement Q-learning algorithm to train the agent to be a better taxi driver. 
# Plot the reward with respect to each episode (during the training and testing phases).
# Plot the number of steps taken with each episode (during the training and testing phases).
def qlearn(max_episodes, qtable, max_steps, epsilon, alpha):
    rewards = []
    steps = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_step = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                action = np.argmax(qtable[state, :])  # Exploitation

            new_state, reward, done, truncated, _ = env.step(action)

            qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state,:]))
            state = new_state
            ep_reward += reward
            ep_step += 1
            if ep_step > max_steps:
                break
        rewards.append(ep_reward)
        steps.append(ep_step)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Print progress
        if episode % (max_episodes // 100) == 0:
            progress = (episode / max_episodes) * 100
            print(f'{progress:.2f}% of Episodes done. Epsilon: {epsilon}')
    return qtable, rewards, steps


def test_qtable(env, qtable, episodes=1, max_steps=100, render=False):
    rewards = []
    steps = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_step = 0
        print(f"Episode {episode + 1}")
        while not done:
            if render:
                env.render()  # Visualize the environment
            action = np.argmax(qtable[state, :])  # Choose the best action based on the Q-table
            new_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_step += 1
            state = new_state
            if ep_step > max_steps:
                break
        print(f"Total reward: {ep_reward}")
        rewards.append(ep_reward)
        steps.append(ep_step)
    if render:
        env.close()
    return rewards, steps


qtable, train_rewards, train_steps = qlearn(train_episodes, qtable, max_steps, epsilon, alpha)
# Create Taxi environment with render_mode for testing
test_env = gym.make("Taxi-v3")
test_rewards, test_steps = test_qtable(test_env, qtable, test_episodes, max_steps)

# Plot rewards and steps per episode
plt.figure(figsize=(12, 5))

# Plot combined rewards
plt.subplot(1, 2, 1)
plt.plot(train_rewards + test_rewards, label='Combined Rewards')  # Plot combined rewards
plt.axvline(x=len(train_rewards), color='r', linestyle='--')  # Dashed red line at the end of training rewards
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.legend()

# Plot steps
plt.subplot(1, 2, 2)
plt.plot(train_steps + test_steps)  # Plot combined steps
plt.axvline(x=len(train_steps), color='r', linestyle='--')  # Dashed red line at the end of training rewards
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.title('Number of Steps per Episode')

plt.tight_layout()
plt.show()
