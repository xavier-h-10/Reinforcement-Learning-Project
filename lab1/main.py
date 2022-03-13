# main.py

import numpy as np
from Gridworld import GridworldEnv
from ValueIteration import value_iteration
from PolicyIteration import policy_iteration

# Grid Map Definition
env_for_value_iter = GridworldEnv([6, 6])
env_for_policy_iter = GridworldEnv([6, 6])


def value_iteration_policy(env):
    policy, V = value_iteration(env, 0.001, 1.0)
    print("------------------------------------")
    print("Value Iteration")
    print("------------------------------------")
    print("Reshaped Policy (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    # Transfer probability distribution into choice
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("------------------------------------")
    print("Final Value Function:")
    print(np.reshape(V, env.shape))
    print("------------------------------------")
    print("")

def policy_iteration_policy(env):
    policy, V = policy_iteration(env, 0.001, 1.0)
    print("------------------------------------")
    print("Policy Iteration")
    print("------------------------------------")
    print("Reshaped Policy (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("------------------------------------")
    print("Final Value function:")
    print(np.reshape(V, env.shape))
    print("------------------------------------")
    print("")


if __name__ == '__main__':
    # Value Iteration
    value_iteration_policy(env_for_value_iter)
    # Policy Iteration
    policy_iteration_policy(env_for_policy_iter)





