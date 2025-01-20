import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

def z_normalize(series):
    """Z-normalize a time series."""
    return (series - np.mean(series)) / np.std(series)

def subsequence_distance(subseq, series):
    """Compute the minimum z-normalized Euclidean distance between a subsequence and a time series."""
    m, n = len(subseq), len(series)
    min_dist = float("inf")
    for i in range(n - m + 1):
        candidate = z_normalize(series[i:i + m])
        dist = np.linalg.norm(subseq - candidate)
        min_dist = min(min_dist, dist)
    return min_dist

def compute_gap(distances, threshold):
    """Calculate the gap score for u-shapelets."""
    DA = distances[distances < threshold]
    DB = distances[distances >= threshold]
    if len(DA) == 0 or len(DB) == 0:
        return -float("inf")  # Invalid split
    gap = np.mean(DB) - np.std(DB) - (np.mean(DA) + np.std(DA))
    return gap

# def extract_u_shapelets(dataset, subseq_len):
#     """
#     Extract u-shapelets from a time series dataset.
#     dataset: List of time series.
#     subseq_len: Length of u-shapelets.
#     """
#     u_shapelets = []
#     remaining_data = dataset.copy()
#     print(len(remaining_data),flush=True)
#     x = len(remaining_data)
#     while len(remaining_data) > 1:
#         if(len(remaining_data) < x/2):
#              print(len(remaining_data),flush=True)
#              x = x/2
            
#         best_gap = -float("inf")
#         best_shapelet = None
#         best_threshold = None
        
#         for series in remaining_data:
#             for i in range(len(series) - subseq_len + 1):
#                 print(f"I = {i} {len(remaining_data)}",flush=True)

#                 candidate = z_normalize(series[i:i + subseq_len])
#                 distances = np.array([subsequence_distance(candidate, s) for s in remaining_data])
#                 sorted_distances = np.sort(distances)
                
#                 # Test thresholds between each pair of distances
#                 for j in range(len(sorted_distances) - 1):
#                     threshold = (sorted_distances[j] + sorted_distances[j + 1]) / 2
#                     gap = compute_gap(distances, threshold)
                    
#                     if gap > best_gap:
#                         best_gap = gap
#                         best_shapelet = candidate
#                         best_threshold = threshold
        
#         if best_shapelet is not None:
#             u_shapelets.append((best_shapelet, best_threshold))
#             distances = np.array([subsequence_distance(best_shapelet, s) for s in remaining_data])
#             remaining_data = [remaining_data[i] for i in range(len(remaining_data)) if distances[i] >= best_threshold]
#         else:
#             break
    
#     return u_shapelets

def extract_u_shapelets(dataset, subseq_len):
    """
    Extract u-shapelets from a time series dataset.
    dataset: List of time series.
    subseq_len: Length of u-shapelets.
    """
    u_shapelets = []
    remaining_data = dataset.copy()  # Start with all time series
    
    while len(remaining_data) > 1:
        best_gap = -float("inf")
        best_shapelet = None
        best_threshold = None
        
        for series in remaining_data:
            for i in range(len(series) - subseq_len + 1):
                print(f"I = {i} {len(remaining_data)}",flush=True)
                candidate = z_normalize(series[i:i + subseq_len])
                distances = np.array([subsequence_distance(candidate, s) for s in remaining_data])
                sorted_distances = np.sort(distances)
                
                # Test thresholds between consecutive distances
                for j in range(len(sorted_distances) - 1):
                    threshold = (sorted_distances[j] + sorted_distances[j + 1]) / 2
                    gap = compute_gap(distances, threshold)
                    
                    if gap > best_gap:
                        best_gap = gap
                        best_shapelet = candidate
                        best_threshold = threshold
        
        if best_shapelet is not None:
            # Add the best u-shapelet and remove explained data
            u_shapelets.append((best_shapelet, best_threshold))
            distances = np.array([subsequence_distance(best_shapelet, s) for s in remaining_data])
            
            # Keep only time series far enough from the u-shapelet
            new_remaining_data = []
            for i in range(len(remaining_data)):
                if distances[i] >= best_threshold:
                    new_remaining_data.append(remaining_data[i])
            
            # Update remaining_data
            remaining_data = new_remaining_data
        else:
            break  # Stop if no valid shapelet is found
    
    return u_shapelets

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
        distances = [subsequence_distance(shapelet, series) for shapelet, _ in u_shapelets]
        distance_map.append(distances)
    
    distance_map = np.array(distance_map)
    
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
train_file = "Gun_Point_TEST.csv"
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
