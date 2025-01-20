import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
def z_normalize(series):
    """Z-normalize a time series."""
    return (series - np.mean(series)) / np.std(series)
def euclideanDistance(subseq, series):
    """Compute the minimum z-normalized Euclidean distance between a subsequence and a time series."""
    m, n = len(subseq), len(series)
    min_dist = float("inf")
    for i in range(n - m + 1):
        candidate = z_normalize(series[i:i + m])
        dist = np.linalg.norm(subseq - candidate)
        min_dist = min(min_dist, dist)
    return min_dist
def computeDistance(s_hat, D):
    """
    Compute the distance vector for a given subsequence and dataset.

    Parameters:
    - s_hat: The candidate subsequence (already z-normalized).
    - D: The dataset (list of time series).

    Returns:
    - dis: A list of distances from the subsequence to all time series in the dataset.
    """
    # Step 1: Initialize the distance list
    dis = []  
    
    # Step 2: Normalize the subsequence (redundant if s_hat is already normalized)
    s_hat = z_normalize(s_hat)
    
    # Step 3: Iterate over all time series in the dataset
    for ts in D:
        dis_i = float('inf')  # Initialize minimum distance for the current time series
        
        # Step 5: Iterate over all possible start positions of subsequences in ts
        for j in range(len(ts) - len(s_hat) + 1):
            # Step 6: Extract and normalize the current subsequence from ts
            z = z_normalize(ts[j:j + len(s_hat)])
            
            # Step 7: Compute the Euclidean distance
            d = euclideanDistance(z, s_hat)
            
            # Step 8: Update the minimum distance for the current time series
            dis_i = min(dis_i, d)
        
        # Step 4: Append the minimum distance for the current time series
        dis.append(dis_i)
    
    # Step 9: Return the distance vector normalized by the square root of the subsequence length
    return np.array(dis) / np.sqrt(len(s_hat))


def extractUShapelets(D, sLen):
    print("START U_SHAPELET",flush=True)

    """
    Extract U-shapelets from the dataset using the provided algorithm.

    Parameters:
    - D: The dataset (list of time series).
    - sLen: The shapelet length.

    Returns:
    - S_hat: The set of U-shapelets.
    """
    S_hat = []  # Initialize the set of U-shapelets

    while len(D) > 1:  # Continue until the dataset has more than one time series
        # Step 2: Initialize candidate u-shapelets
        s_candidates = []
        print("Line 75",flush=True)
        # Step 3: Generate subsequences of length sLen from each time series
        X = 1
        for ts in D:
            print(f"{len(D)} {X} {len(ts)} Line 78",flush=True)
            X +=1
            for sl in range(0, len(ts) - sLen + 1):  # Each subsequence of length sLen
                s_candidates.append(ts[sl:sl + sLen])

        # Step 4: Compute the gap for each candidate u-shapelet
        gaps = []
        print("loop s_candidate",flush=True)
        for s_candidate in s_candidates:
            gap, dt = computeGap(s_candidate, D)  # Use computeGap to get gap score and threshold
            gaps.append((gap, dt, s_candidate))  # Store gap, threshold, and the candidate

        # Step 5: Find the candidate with the maximum gap score
        max_gap, dt, best_shapelet = max(gaps, key=lambda x: x[0])
        S_hat.append(best_shapelet)  # Add the best shapelet to the set

        # Step 6: Compute distances for the selected u-shapelet
        dis = computeDistance(best_shapelet, D)

        # Step 7: Split distances into two groups based on the threshold
        D_L = [D[i] for i in range(len(D)) if dis[i] <= dt]  # Points in D_L
        D_R = [D[i] for i in range(len(D)) if dis[i] > dt]   # Points in D_R

        # Step 8: Stop if only one time series remains in D_L
        print(f"{len(D_L)} LENGHT",flush=True)
        if len(D_L) < 1:
            break

        # Update the dataset to keep only D_R
        D = D_R
    print("END U_SHAPELET",flush=True)
    return S_hat


def computeGap(s_hat, D):
    """
    Compute the gap score for a candidate u-shapelet and dataset.

    Parameters:
    - s_hat: The candidate u-shapelet (z-normalized).
    - D: The dataset (list of time series).

    Returns:
    - maxGap: The maximum gap score.
    - dt: The threshold value that achieves the maximum gap score.
    """
    # Step 1: Compute distance vector
    dis = computeDistance(s_hat, D)
    
    # Step 2: Sort distance vector in ascending order
    dis_sorted = np.sort(dis)
    
    # Step 3: Initialize maxGap and dt
    maxGap = -float("inf")
    dt = 0
    
    # Step 4: Loop through all possible locations of dt
    for i in range(1, len(dis_sorted)):
        # Step 4.1: Compute threshold as midpoint of two consecutive distances
        d_t = (dis_sorted[i - 1] + dis_sorted[i]) / 2
        
        # Step 5: Split distances into two groups based on threshold
        D_L = dis[dis <= d_t]  # Points to the left of d_t
        D_R = dis[dis > d_t]   # Points to the right of d_t
        
        # Ensure both groups are non-empty
        if len(D_L) == 0 or len(D_R) == 0:
            continue
        
        # Step 6: Compute ratio r = |D_L| / |D_R|
        r = len(D_L) / len(D_R)
        if r < 1:
            r = 1 / r
        
        # Skip if ratio exceeds the threshold
        if r > (1 + 1 / 8):
            continue
        
        # Step 7: Compute means and standard deviations
        m_L, m_R = np.mean(D_L), np.mean(D_R)
        s_L, s_R = np.std(D_L), np.std(D_R)
        
        # Step 8: Compute gap score
        gap = abs(m_R - m_L) / (s_L + s_R + 1e-8)  # Add a small constant to avoid division by zero
        
        # Step 9: Update maxGap and dt if gap is larger
        if gap > maxGap:
            maxGap = gap
            dt = d_t

    # Step 10: Return the maximum gap score and the corresponding dt
    return maxGap, dt

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
        distances = [euclideanDistance(shapelet, series) for shapelet, _ in u_shapelets]
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
u_shapelets = extractUShapelets(dataset, subseq_len)
print("line 123",flush=True)
predicted_labels = cluster_with_u_shapelets(dataset, u_shapelets, num_clusters)
print("line 125",flush=True)

# Evaluate clustering performance
rand_index = adjusted_rand_score(true_labels, predicted_labels)
print(f"u-Shapelets: {len(u_shapelets)} extracted.")
print(f"Adjusted Rand Index: {rand_index:.4f}")
# Example dataset
# D = [
#     [1, 2, 3, 4, 5],
#     [2, 3, 4, 5, 6],
#     [10, 11, 12, 13, 14]
# ]

# # Shapelet length
# sLen = 3

# # Extract U-shapelets
# u_shapelets = extractUShapelets(D, sLen)
# print("Extracted U-Shapelets:", u_shapelets)
