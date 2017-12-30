import gym
import sys
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler



class Estimator():
    """
    Value Function approximator.
    """

    def __init__(self, env):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        self.env = env
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        # Used to convert a state to a featurized representation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
                        ("rbf1", RBFSampler(gamma=5.0, n_components=200)),
                        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
        for _ in range(self.env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state]) # Erst observations skalieren
        featurized = self.featurizer.transform(scaled) # Dann in Features transformieren
        return featurized[0] # Transformiere 2D Feature in 400D Feature
        #return scaled[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if a == None:
            pred = []
            for i in range(self.env.action_space.n):
                pred.append(self.models[i].predict([features])[0])
            return pred
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        feature = self.featurize_state(s)
        self.models[a].partial_fit([feature], [y])

        return None


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.0, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        # The policy we're following
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()


        # 1. Generate one step of an episode e
        # 2. Update Q(s,a) according the the reward and next_state of the step
        # 3. Update policy (e-greedy)
        # 4. Coninue with next step

        state = env.reset()
        #print(state)
        #print("#")
        #print(estimator.featurize_state(state))
        #print("###")
        while True:
            probabilities = policy(state)
            action = np.random.choice(np.arange(len(probabilities)),p=probabilities)

            # 1. Generate one step of an episode e
            next_state, reward, done, _ = env.step(action)

            # Update stats for plotting
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] += 1

            # 2. Update Q(s,a) according the the reward and next_state of the step
            q_values_next = estimator.predict(next_state)

            # target
            target = reward + discount_factor*np.max(q_values_next)
            # state = state
            # action = max_action
            estimator.update(state, action, target)

            #Q[state][action] += alpha * (reward + discount_factor*Q[next_state][max_action] - Q[state][action])

            # 3. Update policy (e-greedy)
            # Happens automatically as everytime we call policy() the updated Q value is used for the calculation

            state = next_state

            if done:
                break

            # Episode is over
    return stats
