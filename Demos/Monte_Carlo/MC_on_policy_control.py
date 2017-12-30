import gym
import numpy as np
import sys
import itertools
import random

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")


def sample(totals):
    n = random.uniform(0, totals[-1])
    for i, total in enumerate(totals):
        if n <= total:
            return i

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):

        #nA = len(Q[observation])
        nA = 2
        probs = [epsilon/(nA-1) for i in range(nA)]
        max_action_index = np.argmax(Q[observation])
        probs[max_action_index] = 1-epsilon
        return probs
        
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    #Q = defaultdict(lambda: defaultdict(float))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes+1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = [] # episode = [(state, action, reward),...]

        # Generate an episode
        state = env.reset()
        # observations runden
        state = tuple(round(i,1) for i in state)
        #state = tuple(i for i in state)
        while True:
            probabilities = policy(tuple(round(i,1) for i in state))
            #probabilities = policy(tuple(i for i in state))
            totals = list(itertools.accumulate(probabilities))
            action = sample(totals)

            next_state, reward, done, _ = env.step(action)
            episode.append((tuple(round(i,1) for i in state), reward, action))
            #episode.append((tuple(i for i in state), reward, action))
            state = next_state
            if done:
                #print("Episode {0} wurde nach {1} Zügen beenden".format(i, len(observations)))
                break

        # Episode is over
        #print("Episode: {}".format(episode))
        # Calculate return for each state

        for state_action in set([(step[0], step[2]) for step in episode]):
            state = state_action[0]
            action = state_action[1]
            #print("State: {}".format(state))
            #print("Action: {}".format(action))
            step_index = next(index for index,element in enumerate(episode) \
                               if element[0]==state and element[2]==action)
            G = sum(value[1]*(discount_factor**step) for step, value in enumerate(episode[step_index:]))
            returns_sum[state_action] += G
            returns_count[state_action] += 1.0
            #print(returns_sum[state_action] / returns_count[state_action])
            Q[state][action] = returns_sum[state_action] / returns_count[state_action]
            #print("Q {}".format(Q[state][action]))
            #print("Q {}".format(Q))
    return Q, policy
