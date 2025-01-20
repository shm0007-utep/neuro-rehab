import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
def load_gun_point_dataset(file_path):
    """Load the Gun Point dataset from a CSV file."""
    data = pd.read_csv(file_path, header=None)
    labels = data.iloc[:, 0].values  # First column is the label
    time_series = data.iloc[:, 1:].values  # Remaining columns are the time series
    return time_series, labels

def z_normalize(series):
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return np.zeros_like(series)
    return (series - mean) / std

def euclidean_distance(series1, series2):
    return np.sqrt(np.sum((series1 - series2) ** 2))

def compute_distance(subsequence, dataset):
    distances = []
    subsequence = z_normalize(subsequence)
    for ts in dataset:
        min_distance = float('inf')
        for j in range(len(ts) - len(subsequence) + 1):
            window = ts[j:j + len(subsequence)]
            window = z_normalize(window)
            distance = euclidean_distance(window, subsequence)
            min_distance = min(min_distance, distance)
        distances.append(min_distance)
    return distances

def compute_gap(subsequence, dataset):
    distances = compute_distance(subsequence, dataset)
    distances = np.sort(distances)
    
    max_gap = 0
    optimal_dt = 0
    
    for i in range(1, len(distances)):
        dt = (distances[i - 1] + distances[i]) / 2
        D_A = distances[distances < dt]
        D_B = distances[distances >= dt]
        
        if len(D_A) == 0 or len(D_B) == 0:
            continue
        
        r = len(D_A) / len(D_B)
        k = 2
        if not (1 / k < r < k):
            continue
        
        m_A = np.mean(D_A)
        m_B = np.mean(D_B)
        s_A = np.std(D_A)
        s_B = np.std(D_B)
        
        gap = (m_B - s_B) - (m_A + s_A)
        
        if gap > max_gap:
            max_gap = gap
            optimal_dt = dt
    
    return max_gap, optimal_dt

def extract_u_shapelets(dataset, shapelet_length):
    u_shapelets = []  # Set of u-shapelets, initially empty
    
    for ts in dataset:
        while True:
            cnt = 0  # Count of candidate u-shapelets
            candidate_u_shapelets = []  # Candidate u-shapelets from the current time series
            
            for sl in range(shapelet_length, shapelet_length + 1):  # Loop over shapelet lengths
                for i in range(len(ts) - sl + 1):  # Loop over all subsequences of length `sl`
                    candidate_u_shapelets.append(ts[i:i + sl])
                    cnt += 1
            
            max_gap = 0
            optimal_dt = 0
            best_shapelet_index = 0
            
            for index, candidate in enumerate(candidate_u_shapelets):
                gap, dt = compute_gap(candidate, dataset)
                if gap > max_gap:
                    max_gap = gap
                    optimal_dt = dt
                    best_shapelet_index = index
            
            best_shapelet = candidate_u_shapelets[best_shapelet_index]
            u_shapelets.append(best_shapelet)  # Add the shapelet with the max gap score
            
            distances = compute_distance(best_shapelet, dataset)
            distances = np.array(distances)
            D_A = distances[distances < optimal_dt]
            
            if len(D_A) <= 1:  # Stop if only one cluster remains
                break
            else:
                # Remove instances with distances less than Θ
                D_B = distances[distances >= optimal_dt]
                mean_D_A = np.mean(D_A)
                std_D_A = np.std(D_A)
                theta = mean_D_A + std_D_A
                dataset = [ts for ts in dataset if np.min(compute_distance(best_shapelet, [ts])) >= theta]
    
    return u_shapelets

# # Load Gun Point dataset
# file_path = 'Gun_Point_TRAIN.csv'  # Replace with the actual file path
# time_series, labels = load_gun_point_dataset(file_path)

# # Extract U-Shapelets
# shapelet_length = 30  # Set the shapelet length
# u_shapelets = extract_u_shapelets(time_series, shapelet_length)

# # Print Extracted U-Shapelets
# print("Extracted U-Shapelets:")
# for idx, shapelet in enumerate(u_shapelets):
#     print(f"Shapelet {idx + 1}: {shapelet}")



def cluster_with_u_shapelets(dataset, u_shapelets, num_clusters):
    """
    Cluster time series using the extracted u-shapelets.
    dataset: List of time series.
    u_shapelets: Extracted u-shapelets.
    num_clusters: Number of clusters to find.
    """
    # Compute the distance map
    distance_map = []
    for series in dataset:
        distances = [euclidean_distance(shapelet, series) for shapelet, _ in u_shapelets]
        distance_map.append(distances)
    
    distance_map = np.array(distance_map)
    print("Start K Means",flush=True)
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_map)
    
    return cluster_labels

# Load Gun Point dataset
def load_gun_point_data(train_file, test_file):
    """
    Load the Gun Point dataset from CSV files.
    train_file: Path to the training dataset.
    test_file: Path to the test dataset.
    """
    print("Start")
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    print("READ DATA",flush=True)
    train_labels = train_data.iloc[:, 0].values  # First column is the label
    test_labels = test_data.iloc[:, 0].values  # First column is the label
    
    train_series = train_data.iloc[:, 1:].values  # Remaining columns are time series data
    test_series = test_data.iloc[:, 1:].values  # Remaining columns are time series data
    
    return train_series, train_labels, test_series, test_labels

# Main script

# Paths to Gun Point dataset
train_file = "Gun_Point_TRAIN.csv"
test_file = "Gun_Point_TEST.csv"

# Load dataset
print("START OUTSIDE",flush=True)
train_series, train_labels, test_series, test_labels = load_gun_point_data(train_file, test_file)
dataset = np.vstack((train_series, test_series))  # Combine train and test for clustering
true_labels = np.hstack((train_labels, test_labels))
print("Line 117")
subseq_len = 20  # Length of u-shapelets
num_clusters = 2  # Gun Point has two classes

# Extract u-shapelets and perform clustering
u_shapelets = extract_u_shapelets(dataset, subseq_len)
print("line 123",flush=True)
predicted_labels = cluster_with_u_shapelets(dataset, u_shapelets, num_clusters)
print("line 125",flush=True)

# Evaluate clustering performance
rand_index = adjusted_rand_score(true_labels, predicted_labels)
print(f"u-Shapelets: {len(u_shapelets)} extracted.")
print(f"Adjusted Rand Index: {rand_index:.4f}")