import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load data (assuming one row per series, alternating columns)
df_raw = pd.read_csv('file.csv', sep=' ', header=None)
tidy_list = []

# Reshape data for processing
for row_idx, row in df_raw.iterrows():
    values = row.dropna().values
    ir_vals = values[0::2]
    bright_vals = values[1::2]
    for frame, (ir, bright) in enumerate(zip(ir_vals, bright_vals)):
        tidy_list.append({'Series': row_idx + 1, 'Frame': frame, 'IR': ir, 'Brightness': bright})

df = pd.DataFrame(tidy_list)

# --- FIGURE 1: ORIGINAL OPERATION (COLOR BY SERIES) ---
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
series_ids = df['Series'].unique()
colors = plt.cm.get_cmap('tab10', len(series_ids))

for i, s_id in enumerate(series_ids):
    subset = df[df['Series'] == s_id]
    ax1.scatter(subset['IR'], subset['Brightness'], subset['Frame'], 
                color=colors(i), label=f'Series {s_id}', alpha=0.7)

ax1.set_xlabel('IR Intensity')
ax1.set_ylabel('Brightness')
ax1.set_zlabel('Frame')
ax1.set_title('Plot 1: 3D Scatter by Data Series')
ax1.legend()
plt.show()

# --- K-MEANS CLUSTERING IMPLEMENTATION ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['IR', 'Brightness']])

# Map clusters to binary labels (00, 01, 10, 11)
centers = kmeans.cluster_centers_
mid_ir = np.median(centers[:, 0])
mid_bright = np.median(centers[:, 1])

mapping = {}
for i, center in enumerate(centers):
    # Mode = (Brightness Bit)(IR Bit)
    b_bit = "1" if center[1] > mid_bright else "0"
    i_bit = "1" if center[0] > mid_ir else "0"
    mapping[i] = f"{b_bit}{i_bit}"

df['Mode'] = df['Cluster'].map(mapping)

# --- FIGURE 2: K-MEANS IMPLEMENTATION (COLOR BY MODE) ---
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
mode_colors = {'00': '#1f77b4', '01': '#ff7f0e', '10': '#2ca02c', '11': '#d62728'}

for mode_id, color in mode_colors.items():
    subset = df[df['Mode'] == mode_id]
    ax2.scatter(subset['IR'], subset['Brightness'], subset['Frame'], 
                c=color, label=f'Mode {mode_id}', alpha=0.6)

ax2.set_xlabel('IR Intensity')
ax2.set_ylabel('Brightness')
ax2.set_zlabel('Frame')
ax2.set_title('Plot 2: K-Means Clustering (Operational Modes)')
ax2.legend(title="Binary Mode")
plt.show()

# Save the final results to CSV
df.to_csv('clustered_data.csv', index=False)
