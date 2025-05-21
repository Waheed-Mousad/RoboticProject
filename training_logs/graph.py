import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
csv_path = "model6.csv"  # change path if needed
df = pd.read_csv(csv_path)

# Create plots
plt.figure(figsize=(12, 8))

# Episode vs Episode Score
plt.subplot(3, 1, 1)
plt.plot(df['episode'], df['episode_score'], marker='o')
plt.title('Episode vs Episode Score')
plt.xlabel('Episode')
plt.ylabel('Episode Score')
plt.grid(True)

# Episode vs Average Score
plt.subplot(3, 1, 2)
plt.plot(df['episode'], df['avg_score'], marker='o', color='orange')
plt.title('Episode vs Average Score')
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.grid(True)

# Episode vs Score per Step
plt.subplot(3, 1, 3)
plt.plot(df['episode'], df['score_per_step'], marker='o', color='green')
plt.title('Episode vs Score per Step')
plt.xlabel('Episode')
plt.ylabel('Score per Step')
plt.grid(True)

plt.tight_layout()
plt.show()
