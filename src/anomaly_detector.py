import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

def detect_worker_anomalies(df, eps=1.5, min_samples=3):
    """
    Identifies anomalous worker behavior using PCA and DBSCAN.
    
    Args:
        df (pd.DataFrame): DataFrame containing worker data. 
                           Must include 'mileage', 'journeys', and 'attempts'.
        eps (float): The maximum distance between two samples for one to be 
                     considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point 
                           to be considered as a core point.
                           
    Returns:
        pd.DataFrame: The original DataFrame with added 'pca_1', 'pca_2', 
                      and 'anomaly_label' columns. A label of -1 indicates an anomaly.
    """
    # 1. Feature Engineering: Create ratios to expose behavior
    analysis_df = df.copy()
    
    # Avoid division by zero
    safe_journeys = analysis_df['journeys'].replace(0, 1)
    safe_attempts = analysis_df['attempts'].replace(0, 1)
    
    analysis_df['miles_per_journey'] = analysis_df['mileage'] / safe_journeys
    analysis_df['attempts_per_journey'] = analysis_df['attempts'] / safe_journeys
    
    features = ['mileage', 'journeys', 'attempts', 'miles_per_journey', 'attempts_per_journey']
    X = analysis_df[features]

    # 2. Standardization: Ensure no single metric dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA: Reduce dimensions to capture the primary variance (The 'Map')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    analysis_df['pca_1'] = X_pca[:, 0]
    analysis_df['pca_2'] = X_pca[:, 1]

    # 4. Outlier Detection: DBSCAN isolates workers who break the normal correlations
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    analysis_df['anomaly_label'] = dbscan.fit_predict(X_pca)

    return analysis_df