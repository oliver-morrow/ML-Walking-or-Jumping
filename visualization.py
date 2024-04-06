from main import *
import matplotlib.pyplot as plt

# # PLOT raw data for Oliver from oliver_walking_labeled and oliver_jumping_labeled
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
# oliver_walking_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Oliver Walking (Raw Data)', ax=axes[0])
# oliver_jumping_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Oliver Jumping (Raw Data)', ax=axes[1])
# plt.tight_layout()
# plt.show()

# # PLOT raw data for Matthew from matthew_walking_labeled and matthew_jumping_labeled
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
# matthew_walking_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Matthew Walking (Raw Data)', ax=axes[0])
# matthew_jumping_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Matthew Jumping (Raw Data)', ax=axes[1])
# plt.tight_layout()
# plt.show()

# # PLOT raw data for Daniel from daniel_walking_labeled and daniel_jumping_labeled
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
# daniel_walking_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Daniel Walking (Raw Data)', ax=axes[0])
# daniel_jumping_labeled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Daniel Jumping (Raw Data)', ax=axes[1])
# plt.tight_layout()
# plt.show()

walking_df_notshuffled = pd.concat([oliver_walking_labeled, matthew_walking_labeled, daniel_walking_labeled], ignore_index=True)
jumping_df_notshuffled = pd.concat([oliver_jumping_labeled, matthew_jumping_labeled, daniel_jumping_labeled], ignore_index=True)

# Plot walking_df and jumping_df vs time
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
walking_df_notshuffled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Walking Data', ax=axes[0])
jumping_df_notshuffled.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Jumping Data', ax=axes[1])
plt.tight_layout()
plt.show()

# Apply the moving average filter to walking_df_notshuffled and jumping_df_notshuffled
window_size = 5
walking_df_notshuffled_5 = walking_df_notshuffled.rolling(window_size).mean()
jumping_df_notshuffled_5 = jumping_df_notshuffled.rolling(window_size).mean()

# Plot walking_df_notshuffled_5 and jumping_df_notshuffled_5 vs time
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
walking_df_notshuffled_5.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Walking Data (Moving Average Filtered)', ax=axes[0])
jumping_df_notshuffled_5.plot(x='Time (s)', y=['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)'], title='Jumping Data (Moving Average Filtered)', ax=axes[1])
plt.tight_layout()
plt.show()
