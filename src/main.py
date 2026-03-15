import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detector import detect_worker_anomalies

# 1. Freeze the Data (The "Territory")
def get_frozen_data():
    np.random.seed(42) 
    n_workers = 50
    data = {
        'worker_id': [f'W_{i:02d}' for i in range(n_workers)],
        'mileage': np.random.normal(25, 5, n_workers),
        'journeys': np.random.randint(2, 6, n_workers),
        'attempts': np.random.randint(3, 10, n_workers)
    }
    df = pd.DataFrame(data)
    
    # Inject 2 "Manual" Anomalies for the presentation
    df.loc[48, ['mileage', 'journeys', 'attempts']] = [140.0, 1, 2]   # The "Outlier"
    df.loc[49, ['mileage', 'journeys', 'attempts']] = [15.0, 2, 35]   # The "Inefficient"
    return df

def generate_management_presentation(df):
    # Process data through our detector first
    processed_df = detect_worker_anomalies(df, eps=1.2, min_samples=3)
    
    # --- STAGE 1: HISTOGRAMS ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Stage 1: Distribution of Raw Metrics (Spotting the Tails)", fontsize=16)
    
    cols = ['mileage', 'journeys', 'attempts']
    colors = ['skyblue', 'salmon', 'lightgreen']
    
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i])
        axes[i].set_title(f'Distribution of {col.capitalize()}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- STAGE 2: X/Y CORRELATIONS ---
    plt.figure(figsize=(12, 5))
    plt.suptitle("Stage 2: Operational Correlations (Finding 'Impossible' Behavior)", fontsize=16)
    
    # Mileage vs Journeys
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=processed_df, x='journeys', y='mileage', hue='anomaly_label', 
                    palette={-1: 'red', 0: 'blue', 1: 'blue'}, legend=False)
    plt.title("Mileage vs. Journeys")
    plt.annotate("The 'Gas Guzzler'", xy=(1, 140), xytext=(2, 145),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # Mileage vs Attempts
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=processed_df, x='attempts', y='mileage', hue='anomaly_label', 
                    palette={-1: 'red', 0: 'blue', 1: 'blue'}, legend=False)
    plt.title("Mileage vs. Attempts")
    plt.annotate("The 'Clicker'", xy=(35, 15), xytext=(25, 40),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- STAGE 3: THE PCA MAP ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=processed_df, x='pca_1', y='pca_2', hue='anomaly_label', 
                    style='anomaly_label', palette='coolwarm', s=100)
    
    plt.title("Stage 3: The PCA 'Behavioral Map'\n(Combining all risks into a single view)", fontsize=16)
    plt.xlabel("Primary Behavioral Variance")
    plt.ylabel("Secondary Behavioral Variance")
    plt.legend(title='Status', labels=['Anomaly (Investigate)', 'Normal Cluster'])
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    df = get_frozen_data()
    generate_management_presentation(df)