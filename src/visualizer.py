import matplotlib.pyplot as plt

def plot_worker_anomalies(df):
    """Generates a scatter plot of worker behavior, highlighting anomalies."""
    plt.figure(figsize=(10, 6))
    
    # Separate the honest worker from the anomalous one
    normals = df[df['anomaly_label'] != -1]
    anomalies = df[df['anomaly