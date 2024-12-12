import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
db_path = 'tetris_stats4220.db'
conn = sqlite3.connect(db_path)

query = "SELECT episode, pieces_placed FROM episode_rewards ORDER BY episode;"
data = pd.read_sql_query(query, conn)

conn.close()

txt_file_path = "pieces_placed.txt"
with open(txt_file_path, "w") as file:
    # Write header
    file.write("Game ID\tPieces placed\n")
    file.write("-" * 20 + "\n")
    # Write rows
    for _, row in data.iterrows():
        file.write(f"{row['episode']}\t{row['pieces_placed']}\n")

print(f"Data successfully written to {txt_file_path}")

#data['batch'] = (data['episode'] - 1) //50 +1
# Aggregate the data: sum or mean for each batch
#aggregated_data = data.groupby('batch')['pieces_placed'].sum().reset_index()
#aggregated_data.rename(columns={'batch': 'Batches', 'pieces_placed': 'Lines Cleared'}, inplace=True)


plt.figure(figsize=(12, 6), facecolor='#d3d3d3')


plt.gca().set_facecolor('#a9a9a9')

#plt.bar(aggregated_data['Batches'], aggregated_data['Lines Cleared'],color='#006400' , width=0.8)

plt.scatter(data['episode'], data['pieces_placed'], color='#006400', marker='o', linestyle='-')

# Add labels and title
plt.title("Pieces placed per Game", fontsize=16)
plt.xlabel("Game ID", fontsize=12)
plt.ylabel("Pieces Placed", fontsize=12)

# Adjust x-axis ticks
#plt.xticks(aggregated_data['Batches'],
      #     [f"{(batch-1)*50+1}-{batch*50}" for batch in aggregated_data['Batches']],
       #    rotation=45)
plt.ylim(0, data['pieces_placed'].max() + 10)  # Extend the upper limit of the y-axis
plt.yticks(range(0, int(data['pieces_placed'].max()) + 10, 10))

plt.xticks(rotation=45)
plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

# Display the plot
plt.tight_layout()
plt.savefig("pieces_placed_chart.png", dpi=300, bbox_inches='tight')
plt.show()