import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import os

def generate_coverage_map(predictions_path, output_dir):
    """Generate coverage maps from prediction results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    print(f"Loading predictions from {predictions_path}")
    grid_stats = pd.read_csv(predictions_path)
    
    # Extract coordinates
    min_lat = grid_stats['lat_grid'].min()
    max_lat = grid_stats['lat_grid'].max()
    min_lon = grid_stats['lon_grid'].min()
    max_lon = grid_stats['lon_grid'].max()
    
    # Create grid for visualization
    lat_step = (max_lat - min_lat) / 100
    lon_step = (max_lon - min_lon) / 100
    
    lat_grid = np.arange(min_lat, max_lat, lat_step)
    lon_grid = np.arange(min_lon, max_lon, lon_step)
    
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    prediction_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
    
    # Use KNN for interpolation
    X_spatial = grid_stats[['lat_grid', 'lon_grid']].values
    
    # Plot RSSI heatmap
    plt.figure(figsize=(12, 10))
    rssi_map = plt.scatter(
        grid_stats['lon_grid'],
        grid_stats['lat_grid'],
        c=grid_stats['rssi_mean'],
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='k'
    )
    plt.colorbar(rssi_map, label='RSSI (dBm)')
    plt.title('Average RSSI by Grid Cell')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/rssi_heatmap.png", dpi=300)
    plt.close()
    
    # Create coverage prediction map
    plt.figure(figsize=(12, 10))
    
    # Interpolate coverage predictions
    knn = KNeighborsRegressor(n_neighbors=5)
    y_spatial = grid_stats['ensemble_prob'].values
    knn.fit(X_spatial, y_spatial)
    
    # Make predictions
    grid_predictions = knn.predict(prediction_points)
    grid_predictions = grid_predictions.reshape(lon_mesh.shape)
    
    # Plot interpolated coverage
    coverage_map = plt.contourf(
        lon_mesh, lat_mesh, grid_predictions, 
        levels=50, cmap='RdYlGn_r', alpha=0.7
    )
    plt.colorbar(coverage_map, label='Low Coverage Probability')
    
    # Plot predicted grid cells
    low_coverage_points = grid_stats[grid_stats['ensemble_prediction'] == 1]
    plt.scatter(
        low_coverage_points['lon_grid'],
        low_coverage_points['lat_grid'],
        c='red', marker='x', s=30,
        label='Predicted Low Coverage'
    )
    
    # Add details
    plt.title('Predicted Low Coverage Areas Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/coverage_prediction_map.png", dpi=300)
    plt.close()
    
    # Create anomaly map if we have anomaly data
    if 'anomaly_density' in grid_stats.columns:
        plt.figure(figsize=(12, 10))
        
        # Create a continuous colormap of anomaly density
        anomaly_map = plt.scatter(
            grid_stats['lon_grid'],
            grid_stats['lat_grid'],
            c=grid_stats['anomaly_density'],
            cmap='plasma',
            alpha=0.7,
            s=50,
            edgecolors='k'
        )
        plt.colorbar(anomaly_map, label='Anomaly Density')
        
        # Overlay low coverage contours
        contour = plt.contour(
            lon_mesh, 
            lat_mesh, 
            grid_predictions, 
            levels=[0.5], 
            colors='red', 
            linewidths=2
        )
        plt.clabel(contour, inline=True, fontsize=10, fmt='Low Coverage')
        
        plt.title('Signal Anomalies and Coverage Problems')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/anomaly_coverage_map.png", dpi=300)
        plt.close()
    
    # Compare model predictions 
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(grid_stats['rssi_mean'], grid_stats['rf_prob'], alpha=0.3)
    plt.title('Random Forest Probability vs RSSI')
    plt.xlabel('RSSI Mean (dBm)')
    plt.ylabel('Low Coverage Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(grid_stats['rssi_mean'], grid_stats['gb_prob'], alpha=0.3)
    plt.title('Gradient Boosting Probability vs RSSI')
    plt.xlabel('RSSI Mean (dBm)')
    plt.ylabel('Low Coverage Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(grid_stats['rssi_std'], grid_stats['rf_prob'], alpha=0.3)
    plt.title('Random Forest Probability vs RSSI Std')
    plt.xlabel('RSSI Standard Deviation')
    plt.ylabel('Low Coverage Probability')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.scatter(grid_stats['bssid_count'], grid_stats['rf_prob'], alpha=0.3)
    plt.title('Random Forest Probability vs BSSID Count')
    plt.xlabel('BSSID Count per Grid')
    plt.ylabel('Low Coverage Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_analysis.png", dpi=300)
    plt.close()
    
    print(f"Generated plots saved to {output_dir}")

if __name__ == "__main__":
    generate_coverage_map(
        predictions_path="output_final/coverage_predictions.csv",
        output_dir="plots_final"
    ) 