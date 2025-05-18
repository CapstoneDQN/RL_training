
# baseline_rl_training.py
# Tabular Q-learning baseline using only listening_history.csv
# Enhanced logging: episode summaries and sample recommendations

import pandas as pd
import numpy as np
import random
import pickle

# Hyperparameters
ALPHA = 0.1        # learning rate
GAMMA = 0.9        # discount factor
EPSILON = 0.1      # exploration rate
EPISODES = 10      # number of passes over all user histories
TOP_K = 500        # consider only top-K frequent songs

# 1) Load whitespace-delimited listening_history.csv and parse timestamp
df = pd.read_csv(
    'listening_history.csv', sep=r'\s+', header=None,
    names=['user','song','date','time']
)
df['timestamp'] = df['date'] + ' ' + df['time']

# 2) Limit to top-K frequent songs
song_counts = df['song'].value_counts()
top_items = song_counts.head(TOP_K).index.tolist()
df = df[df['song'].isin(top_items)]

# 3) Sort by user & timestamp, then aggregate songs per user
df = df.sort_values(['user','timestamp'])
user_groups = df.groupby('user')['song'].apply(list).reset_index()
user_groups.columns = ['user_id','listening_history']

# 4) Build item-index mappings for top-K
item2idx = {item: idx for idx, item in enumerate(top_items)}
idx2item = {idx: item for item, idx in item2idx.items()}
num_items = len(top_items)

# 5) Initialize Q-table (TOP_K x TOP_K)
Q = np.zeros((num_items, num_items), dtype=np.float32)

# 6) Q-learning training with episode-level logging and reward tracking
episode_rewards = []
def train_q_learning(user_histories, Q_table):
    for ep in range(1, EPISODES + 1):
        total_reward = 0.0
        total_steps = 0
        for hist in user_histories:
            hist = [s for s in hist if s in item2idx]
            if len(hist) < 2:
                continue
            for i in range(1, len(hist)):
                state_idx = item2idx[hist[i - 1]]
                true_idx = item2idx[hist[i]]
                if random.random() < EPSILON:
                    action = random.randrange(num_items)
                else:
                    action = np.argmax(Q_table[state_idx])
                reward = 1.0 if action == true_idx else 0.0
                best_next = np.max(Q_table[true_idx])
                Q_table[state_idx, action] += ALPHA * (reward + GAMMA * best_next - Q_table[state_idx, action])
                total_reward += reward
                total_steps += 1
        avg_reward = total_reward / total_steps if total_steps else 0.0
        episode_rewards.append(avg_reward)
        print(f"Episode {ep}/{EPISODES} - Total Reward: {total_reward:.2f}, Avg Reward per Step: {avg_reward:.4f}")

# Run training
train_q_learning(user_groups['listening_history'], Q)

# Summary of training rewards
print("\nTraining Summary:")
for i, r in enumerate(episode_rewards, start=1):
    print(f"  Episode {i}: Avg Reward {r:.4f}")
print(f"  Overall Avg Reward: {np.mean(episode_rewards):.4f} (std {np.std(episode_rewards):.4f})")

# 7) Policy evaluation: calculate accuracy
def evaluate(user_histories, Q_table):
    correct, total = 0, 0
    for hist in user_histories:
        hist = [s for s in hist if s in item2idx]
        if len(hist) < 2:
            continue
        for i in range(1, len(hist)):
            state_idx = item2idx[hist[i - 1]]
            true_idx = item2idx[hist[i]]
            pred = np.argmax(Q_table[state_idx])
            correct += int(pred == true_idx)
            total += 1
    return correct / total if total else 0.0

accuracy = evaluate(user_groups['listening_history'], Q)
print("\nEvaluation Results:")
print(f"  Q-learning Baseline Accuracy (Top-{TOP_K} items): {accuracy:.4f}")

# 8) Sample recommendations for popular states
print("\nSample Recommendations for Top-5 Songs:")
for song in top_items[:5]:
    s_idx = item2idx[song]
    top_actions = np.argsort(-Q[s_idx])[:5]
    rec_songs = [idx2item[a] for a in top_actions]
    print(f"  After '{song}': {rec_songs}")

# 9) Save Q-table
with open('q_table.pkl', 'wb') as f:
    pickle.dump({'Q': Q, 'item2idx': item2idx, 'idx2item': idx2item}, f)
print(f"\nSaved Q-table for top-{TOP_K} items to q_table.pkl")
