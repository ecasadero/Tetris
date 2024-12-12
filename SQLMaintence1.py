import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
db_path = 'tetris_stats4220.db'
conn = sqlite3.connect(db_path)

# Query the data
query = "SELECT game_id, total_lines FROM game_stats ORDER BY game_id;"
data = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Create batches
data['batch'] = (data['game_id'] - 1) // 50 + 1

# Aggregate the data: sum for each batch
aggregated_data = data.groupby('batch')['total_lines'].sum().reset_index()
aggregated_data.rename(columns={'batch': 'Batches', 'total_lines': 'Lines Cleared'}, inplace=True)

# Plot the bar graph
plt.figure(figsize=(12, 6), facecolor='#f5f5f5')  # Set figure background here

# Set the plot background color
plt.gca().set_facecolor('#f5f5f5')

# Plot the bars with a soft dark green color
plt.bar(aggregated_data['Batches'], aggregated_data['Lines Cleared'], color='#006400', width=0.8)

# Add labels and title
plt.title("Lines Cleared by Batches of 50 Games", fontsize=16)
plt.xlabel("Batch (Game ID Ranges)", fontsize=12)
plt.ylabel("Lines Cleared", fontsize=12)

# Adjust x-axis ticks
plt.xticks(
    aggregated_data['Batches'],
    [f"{(batch-1)*50+1}-{batch*50}" for batch in aggregated_data['Batches']],
    rotation=45
)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
