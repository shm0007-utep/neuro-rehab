import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read the accelerometer data from the CSV file
file_path = 'data/demo.csv'
df = pd.read_csv(file_path)

# Assuming the columns are named 'time', 'x', 'y', and 'z'
time = df['Timestamp UTC']
x = df['Accelerometer X']
y = df['Accelerometer Y']
z = df['Accelerometer Z']
# Create a plot for the accelerometer data
plt.figure(figsize=(10, 6))
plt.plot(time, x, label='X-axis', color='red')
plt.plot(time, y, label='Y-axis', color='green')
plt.plot(time, z, label='Z-axis', color='blue')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Accelerometer Data (X, Y, Z) vs. Time')
plt.legend()

# Show the plot
plt.show()