import numpy as np
def load_ts_file(file_path):
    """Load a .ts file and return the data and labels."""
    data = []
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        reading_data = False
        for line in lines:
            line = line.strip()
            if line.startswith('@data'):
                reading_data = True
                continue
            if reading_data:
                parts = line.split(',')
                labels.append(int(parts[0]))  # First value is the label
                data.append([float(x) for x in parts[1:]])  # Remaining are the data points
    return np.array(data), np.array(labels)

# Example usage
file_path = 'GunPoint_TRAIN.ts'
data, labels = load_ts_file(file_path)
print("Data shape:", data.shape)
print("Labels:", labels)
