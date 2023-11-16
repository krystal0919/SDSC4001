from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gym

env = gym.make("Blackjack-v1", sab=True)
Is_done = False
observation, info = env.reset()
env.observation_space
env.action_space.n
print(env.observation_space.sample())
print(env.player)
print(env.dealer)

discount_factor = 0
learning_rate = 0.9
Q = np.zeros((32, 11, 2, 2))
print(Q.shape)  # Print the shape of Q-table

num_episodes = 10
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

for episode in range(num_episodes):
    state = env.reset()

    Is_done = False
    current_rewards = 0
    
    print("state:", state)  # Print the state tuple
    
    player_sum = state[0][0]
    dealer_card = state[0][1]  # Access the first element of the dealer_card list
    
    print("dealer_card:", dealer_card)
    
    if np.random.uniform(0, 1) > exploration_rate:
        action = np.argmax(Q[player_sum, dealer_card])
    else:
        action = env.action_space.sample()
        
    next_state, reward, Is_done, _, _ = env.step(action)

    print("player_sum:", player_sum)
    print("dealer_card:", dealer_card)
    print("action:", action)
    print("next_state[0]:", next_state[0])
    print("next_state[1]:", next_state[1])

    Q[player_sum, dealer_card, action] = (1 - learning_rate) * Q[player_sum, dealer_card, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state[0], next_state[1]]))
    
    state = next_state
    current_rewards += reward

    if Is_done:
        break
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)