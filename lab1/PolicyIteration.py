import numpy as np


# Calculate the value for all actions in a given state, one-step lookahead
# state: state (int)
# V: value function, vector with the length of env.nS
# _discount_factor: discount factor
# my_policy: current policy (FIXED in Policy Evaluation part)
# Return: a 
def calculate_action_value(env, state, V, _discount_factor, my_policy):
    A = 0
    for action in range(env.nA):
        for prob, next_state, reward, isDone in env.P[state][action]:
            A += my_policy[state][action] * prob * (reward + _discount_factor * V[next_state])
    return A


# Greedy policy
def greedy_policy(env, state, V, _discount_factor):
    A = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, isDone in env.P[state][action]:
            A[action] += prob * (reward + _discount_factor * V[next_state])             # Equals to: A[action] = V[next_state]
    maxElm = -100000000.0
    # Note that the actions corresponding to the maximum A may be more than one!
    ret = []
    for action in range(env.nA):
        if A[action] > maxElm:
            maxElm = A[action]
            ret.clear()
            ret.append(action)
        elif A[action] == maxElm:
            ret.append(action)
    return ret


# env: gridworld
# _theta: Stopping threshold
# _discount_factor: discount factor
# Return: a tuple (policy, V) of the optimal policy and the optimal value function
def policy_iteration(env, _theta=0.001, _discount_factor=1.0):
    # Value function
    V = np.zeros(env.nS)
    # Iteration step
    iteration_step = 0
    # Update time for value function
    update_times_for_value_function = 0
    # Create a deterministic policy using the optimal value function.
    my_policy = np.zeros([env.nS, env.nA])
    # Policy Initialization. Policy is initialized to be 0.25 for all 4 directions
    for state in range(env.nS):
        for action in range(env.nA):
            my_policy[state][action] = 0.25

    while True:
        iteration_step += 1

        # Policy Evaluation, Update value function in current policy
        while True:
            update_times_for_value_function += 1
            # Stop condition
            _delta = 0
            # Update each state
            for state in range(env.nS):
                origin_value = V[state]
                # Update value function in current policy
                V[state] = calculate_action_value(env, state, V, _discount_factor, my_policy)
                # Calculate _delta across all states seen so far
                _delta = max(_delta, np.abs(origin_value - V[state]))
            if _delta < _theta:
                break

        # Policy Improvement
        policy_stable = True  # Whether this policy is stable
        for state in range(env.nS):
            # Get the old action (with the largest probability in my_policy[state])
            maxElm = -100000000.0
            old_actions = []
            for action in range(env.nA):
                if my_policy[state][action] > maxElm:
                    maxElm = my_policy[state][action]
                    old_actions.clear()
                    old_actions.append(action)
                elif my_policy[state][action] == maxElm:
                    old_actions.append(action)

            # Get the currently best action by using greedy search
            best_actions = greedy_policy(env, state, V, _discount_factor)
            # Policy is changed in this update round
            # Note that the policy grid may be more than one direction!
            if len(old_actions) != len(best_actions):
                policy_stable = False
            else:
                for i in range(0, len(best_actions), 1):
                    if best_actions[i] not in old_actions:
                        policy_stable = False
                        break

            if not policy_stable:
                prob = 1.0 / float(len(best_actions))
                for action in range(env.nA):
                    my_policy[state][action] = 0.0
                for i in range(0, len(best_actions), 1):
                    my_policy[state][best_actions[i]] = prob

        # Stable
        if policy_stable:
            print("Stop. Totally", iteration_step, "iterations for Policy Iteration,", update_times_for_value_function, "times for the update of value function")
            return my_policy, V
