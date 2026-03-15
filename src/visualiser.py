import matplotlib.pyplot as plt

def plot_worker_anomalies(df):
    """Generates a scatter plot of worker behavior, highlighting anomalies."""
    plt.figure(figsize=(10, 6))
    
    # Separate the honest worker from the anomalous one
    # DBSCAN labels anomalies as -1
    normals = df[df['anomaly_label'] != -1]
    anomalies = df[df['anomaly_label'] == -1]
    
    # Plot normal behavior
    plt.scatter(normals['pca_1'], normals['pca_2'], 
                c='blue', label='Normal Behavior', alpha=0.6, edgecolors='w', s=80)
    
    # Plot anomalies (fraud/inefficiency)
    plt.scatter(anomalies['pca_1'], anomalies['pca_2'], 
                c='red', label='Anomaly', alpha=0.8, edgecolors='k', s=100, marker='X')
    
    # Add labels and formatting
    plt.title("Worker Behavior: PCA & DBSCAN Anomaly Detection")
    plt.xlabel("Principal Component 1 (Variance Driver)")
    plt.ylabel("Principal Component 2 (Secondary Traits)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Finalize and show
    plt.tight_layout()
    plt.show()

# Example usage (if running this file directly):
if __name__ == "__main__":
    print("Visualiser module ready. Pass a processed DataFrame to plot_worker_anomalies().")