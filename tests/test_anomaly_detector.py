import pandas as pd
import pytest
from src.anomaly_detector import detect_worker_anomalies

@pytest.fixture
def sample_worker_data():
    """
    Creates synthetic data with a clear cluster of normal behavior
    and one distinct anomaly (Worker E).
    """
    data = {
        'worker_id': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        # Normal workers do ~5-10 miles, 2-3 journeys, 2-4 attempts
        # Worker E claims 100 miles for only 1 journey and 1 attempt (Highly anomalous)
        'mileage': [10.5, 8.0, 12.1, 9.5, 150.0, 11.0, 7.5],
        'journeys': [3, 2, 3, 2, 1, 3, 2],
        'attempts': [4, 2, 5, 3, 1, 4, 3]
    }
    return pd.DataFrame(data)


def test_anomaly_detection_identifies_outlier(sample_worker_data):
    # Act
    result_df = detect_worker_anomalies(sample_worker_data, eps=1.5, min_samples=2)
    
    # Assert
    # Check that the expected columns were added
    assert 'pca_1' in result_df.columns
    assert 'pca_2' in result_df.columns
    assert 'anomaly_label' in result_df.columns
    
    # Extract the anomaly label for Worker E
    worker_e_label = result_df.loc[result_df['worker_id'] == 'E', 'anomaly_label'].values[0]
    
    # In DBSCAN, -1 indicates noise/anomaly. 
    # We expect Worker E, given his inflated mileage, to be flagged as -1.
    assert worker_e_label == -1

def test_normal_workers_are_clustered_together(sample_worker_data):
    # Act
    result_df = detect_worker_anomalies(sample_worker_data, eps=1.5, min_samples=2)
    
    # Extract labels for the normal workers
    normal_workers = result_df[result_df['worker_id'] != 'E']
    unique_labels = normal_workers['anomaly_label'].unique()
    
    # Assert that none of the normal workers are flagged as anomalies (-1)
    assert -1 not in unique_labels
