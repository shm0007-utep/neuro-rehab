import numpy as np

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

# Example Data
dataset = [
    np.array([1, 2, 3, 4, 5, 6]),
    np.array([2, 3, 4, 5, 6, 7]),
    np.array([3, 4, 5, 6, 7, 8])
]

shapelet_length = 3
u_shapelets = extract_u_shapelets(dataset, shapelet_length)

print("Extracted U-Shapelets:")
for idx, shapelet in enumerate(u_shapelets):
    print(f"Shapelet {idx + 1}: {shapelet}")
