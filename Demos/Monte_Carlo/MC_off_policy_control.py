import gym
import matplotlib
import numpy as np
import sys
import itertools
import random

from collections import defaultdict
from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")

def sample(totals):
    n = random.uniform(0, totals[-1])
    for i, total in enumerate(totals):
        if n <= total:
            return i

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn


def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.
    
    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """ 
    def policy_fn(observation):
        A = np.zeros(len(Q[observation]), dtype=float)
        max_action = np.argmax(Q[observation])
        A[max_action] = 1
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Lambda discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Dactionary that stores the commulative sum of the weights for each state action pair
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
    
    for n in range(1,num_episodes+1):
        if n % 1000 == 0:
            print("\rEpisodes: {0}/{1}".format(n,num_episodes),end="")
        
        # Store all steps of each generated episode in an array
        episode = []
        
        # Generate an episode using the behavior policy
        state = env.reset()
        state = tuple(round(i,1) for i in state)
        
        while True:
            probabilities = behavior_policy(tuple(round(i,1) for i in state))
            totals = list(itertools.accumulate(probabilities))
            action = sample(totals)
                         
            next_state, reward, done, _ = env.step(action)
            episode.append((tuple(round(i,1) for i in state), action, reward))
            state = next_state
            
            if done:
                #print("Episode {0} wurde nach {1} Zügen beenden".format(i, len(observations)))
                break
             
            # Episode is over
       
        #print("Laenge von Episode: {}".format(len(episode[n]))))
        
        G = 0.0
        W = 1.0
            
        # Go through the episode backwards
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]

            G = discount_factor * G + reward
            #print("G: {}".format(G))

            # Update cuumulative sum of weights for each state action pair
            C[state][action] += W

            # Update action value function
            Q[state][action] += (W/C[state][action]) * (G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break

            # Update weights
            W = W * 1./behavior_policy(state)[action]
            #             print("G: {}".format(G))
            #             print("C: {}".format(C))
            #             print("Episode: {}".format(episode))
               
    return Q, target_policy
