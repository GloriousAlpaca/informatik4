import matplotlib.pyplot as plt
import numpy as np

# Define Environment
enviro = np.full((3, 3), -1)
end_goal = (0, 2)
# Define Endgoal
enviro[0, 2] = 10
# Define r
enviro[0, 0] = 3
gamma = 0.5

utilities = enviro.copy()

# Possible Actions in the environment
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Probability of Actions
prob = [0.8, 0.1, 0.1]

model = [actions, prob]


def get_next_state(current_state, action):
    i, j = current_state
    di, dj = action
    next_i, next_j = i + di, j + dj
    if 0 <= next_i < 3 and 0 <= next_j < 3:
        return (next_i, next_j)
    return (i, j)  # Stay in the same state if would move out of bounds


def value_iteration(env, iterations, model, gamma):
    utils_per_it = [env.astype(float)]  # Ensure the initial utilities are floats
    for iter in range(iterations):
        iteration_util = np.zeros_like(env, dtype=float)  # Explicitly use float type
        for i in range(len(env)):
            for j in range(len(env[i])):
                if (i, j) == end_goal:
                    iteration_util[i][j] = env[i][j]
                    continue

                a = []
                for action in model[0]:
                    sum_action = 0.0

                    # Intended action
                    next_state = get_next_state((i, j), action)
                    sum_action += model[1][0] * utils_per_it[iter][next_state]

                    # Left of intended
                    left_action = (-action[1], action[0])  # Rotate left
                    next_state = get_next_state((i, j), left_action)
                    sum_action += model[1][1] * utils_per_it[iter][next_state]

                    # Right of intended
                    right_action = (action[1], -action[0])  # Rotate right
                    next_state = get_next_state((i, j), right_action)
                    sum_action += model[1][2] * utils_per_it[iter][next_state]
                    a.append(sum_action)

                iteration_util[i][j] = env[i][j] + gamma * max(a)
        utils_per_it.append(iteration_util)
    return utils_per_it


def plot_utility_estimates(utils_per_it, states_to_plot):
    plt.figure(figsize=(10, 6))

    iterations = range(len(utils_per_it))

    # Store plot lines, labels, and their final utilities
    lines = []
    labels = []
    final_utilities = []

    for state in states_to_plot:
        i, j = state
        label = f"({3 - i},{j + 1})"  # Corrected labeling
        utilities = [u[i, j] for u in utils_per_it]
        line, = plt.plot(iterations, utilities)
        lines.append(line)
        labels.append(label)
        final_utilities.append(utilities[-1])  # Store the final utility value

    plt.xlabel('Number of iterations')
    plt.ylabel('Utility estimates')
    plt.title('Utility Estimates vs. Iterations')
    plt.grid(True)

    # Sort the labels and lines based on final utility values (in descending order)
    sorted_items = sorted(zip(final_utilities, labels, lines), reverse=True)
    sorted_labels = [label for _, label, _ in sorted_items]
    sorted_lines = [line for _, _, line in sorted_items]

    # Add sorted legend
    plt.legend(sorted_lines, sorted_labels)

    plt.show()


def get_policy(utilities, model):
    policy = np.full((3, 3), '', dtype=object)
    for i in range(3):
        for j in range(3):
            if (i, j) == end_goal:  # End goal
                policy[i, j] = 'G'
                continue
            best_action = None
            max_utility = float('-inf')
            for action in model[0]:
                next_state = get_next_state((i, j), action)
                if utilities[next_state] > max_utility:
                    max_utility = utilities[next_state]
                    best_action = action
            if best_action == (-1, 0):
                policy[i, j] = 'U'
            elif best_action == (0, -1):
                policy[i, j] = 'D'
            elif best_action == (0, -1):
                policy[i, j] = 'L'
            elif best_action == (0, 1):
                policy[i, j] = 'R'
    return policy


# Run for r = +3
enviro[0, 0] = 3
utils_per_it_plus3 = value_iteration(enviro, 30, model, gamma)

print("Utilities after 2 iterations for r=+3:")
print(utils_per_it_plus3[1])
print(utils_per_it_plus3[2])

print("\nFinal (converged) utilities for r=+3:")
print(utils_per_it_plus3[-1])

policy_plus3 = get_policy(utils_per_it_plus3[-1], model)
print("\nPolicy for r=+3:")
print(policy_plus3)

# Plot the utilities for r=+3
states_to_plot = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
plot_utility_estimates(utils_per_it_plus3, states_to_plot)

# Run for r = -3
enviro[0, 0] = -3
utils_per_it_minus3 = value_iteration(enviro, 30, model, gamma)

print("\nUtilities after 2 iterations for r=-3:")
print(utils_per_it_minus3[1])
print(utils_per_it_minus3[2])

print("\nFinal (converged) utilities for r=-3:")
print(utils_per_it_minus3[-1])

policy_minus3 = get_policy(utils_per_it_minus3[-1], model)
print("\nPolicy for r=-3:")
print(policy_minus3)

# Plot the utilities for r=-3
plot_utility_estimates(utils_per_it_minus3, states_to_plot)
