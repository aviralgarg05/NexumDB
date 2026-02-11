import matplotlib.pyplot as plt
import numpy as np

# Data from our bench run
labels = ['INSERT', 'SELECT (Cold)', 'SELECT (Cached)']
sqlite_times = [15.18, 0.143, 0.143] # converted to ms
nexum_times = [7.48, 1.86, 1.87]    # in ms

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, sqlite_times, width, label='SQLite', color='#3498db')
rects2 = ax.bar(x + width/2, nexum_times, width, label='NexumDB', color='#e74c3c')

ax.set_ylabel('Latency (ms) - Lower is Better')
ax.set_title('NexumDB vs SQLite Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Log scale helps see the SELECT differences more clearly
ax.set_yscale('log') 

plt.tight_layout()
plt.savefig('bench_results.png')
print("Chart saved as bench_results.png")