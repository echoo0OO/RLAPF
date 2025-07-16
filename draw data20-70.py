import matplotlib.pyplot as plt

# Your provided data
data_amount = [20, 30, 40, 50, 60, 70]
average_steps = [142.05, 157.00, 175.85, 186.25, 192.65, 210.00]

# Assuming 1 step = 1 second for converting steps to time
time_in_seconds = [steps for steps in average_steps] # Direct conversion

# Create the plot
plt.figure(figsize=(10, 6)) # Set the figure size for better readability
plt.plot(data_amount, time_in_seconds, marker='o', linestyle='-', color='b') # Plot with markers and a line

# Add labels and title in English
plt.xlabel('Data Amount/Mb')
plt.ylabel('Time/s')
plt.title('Relationship between Data Amount and Required Time')
plt.grid(True) # Add a grid for easier reading of values

# Add text labels for each point (optional, but helpful for precise values)
for i, txt in enumerate(time_in_seconds):
    plt.annotate(f'{txt:.2f}', (data_amount[i], time_in_seconds[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Show the plot
plt.show()