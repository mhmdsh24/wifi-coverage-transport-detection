import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class EnhancedCoverageVisualizer:
    """
    Enhanced WiFi Coverage Visualization
    
    This class creates enhanced visualizations for WiFi coverage analysis,
    including sample density, confidence levels, and transport mode filtering.
    """
    
    def __init__(self, output_dir='output', plots_dir='plots'):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        plots_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(f"{plots_dir}/enhanced_maps", exist_ok=True)
        
        # Define color schemes with colorblind-safe options
        self.viridis = cm.get_cmap('viridis')
        self.plasma = cm.get_cmap('plasma')
        self.cividis = cm.get_cmap('cividis')  # More colorblind-friendly
        
        # Custom colormap for coverage (green-yellow-red)
        self.coverage_cmap = LinearSegmentedColormap.from_list(
            'coverage_cmap', ['#1a9850', '#ffffbf', '#d73027']
        )
    
    def create_density_overlay_map(self, grid_stats, center=None, zoom_start=13):
        """
        Create a map with coverage and sample density overlay
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with coverage and location data
        center : tuple, optional
            (latitude, longitude) center of the map
        zoom_start : int, default=13
            Initial zoom level
            
        Returns:
        --------
        m : folium.Map
            Folium map object
        """
        print("Creating coverage map with sample density overlay...")
        
        # Ensure required columns are present
        required_cols = ['lat_grid', 'lon_grid', 'rssi_mean', 'rssi_count']
        if not all(col in grid_stats.columns for col in required_cols):
            missing = [col for col in required_cols if col not in grid_stats.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute center if not provided
        if center is None:
            center = (
                grid_stats['lat_grid'].mean(),
                grid_stats['lon_grid'].mean()
            )
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        folium.TileLayer('Stamen Terrain', name='Terrain Map').add_to(m)
        
        # Prepare data for coverage visualization
        # Normalize RSSI values for coloring (-85 to -65 dBm)
        grid_stats['rssi_norm'] = np.clip(
            (grid_stats['rssi_mean'] - (-85)) / ((-65) - (-85)),
            0, 1
        )
        
        # Normalize sample count for transparency
        max_count = grid_stats['rssi_count'].max()
        min_count = grid_stats['rssi_count'].min()
        count_range = max_count - min_count
        grid_stats['opacity'] = np.clip(
            (grid_stats['rssi_count'] - min_count) / (count_range if count_range > 0 else 1),
            0.2, 0.9  # Min 20% opacity, max 90%
        )
        
        # Add coverage layer with density-based opacity
        coverage_layer = folium.FeatureGroup(name="Coverage with Sample Density")
        
        for _, row in grid_stats.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                continue
                
            # Get color based on normalized RSSI - use cividis colormap (colorblind-friendly)
            color = self.cividis(1 - row['rssi_norm'])  # Invert for blue=low, yellow=high in cividis
            html_color = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {row['opacity']})"
            
            # Create enhanced tooltip with detailed information
            tooltip = (
                f"<div style='font-family: Arial; font-size: 12px;'>"
                f"<b>RSSI:</b> {row['rssi_mean']:.1f} dBm<br>"
                f"<b>Sample Count:</b> {row['rssi_count']}<br>"
                f"<b>Std Dev:</b> {row['rssi_std']:.2f}<br>"
            )
            
            # Add SSID info if available
            if 'ssid' in row:
                tooltip += f"<b>SSID:</b> {row['ssid']}<br>"
            
            # Add channel info if available
            if 'channel' in row:
                tooltip += f"<b>Channel:</b> {row['channel']}<br>"
                
            # Add transport mode if available
            if 'predicted_mode' in row:
                tooltip += f"<b>Transport Mode:</b> {row['predicted_mode']}<br>"
                
            # Add anomaly info if available
            if 'is_anomaly_mean' in row and row['is_anomaly_mean'] > 0.2:
                tooltip += f"<b>Anomaly Score:</b> {row['is_anomaly_mean']:.2f} <span style='color:red'>(Warning!)</span><br>"
                
            # Add predicted coverage probability if available
            if 'coverage_probability' in row:
                confidence = abs(row['coverage_probability'] - 0.5) * 2  # 0 to 1 scale
                tooltip += f"<b>Coverage Probability:</b> {row['coverage_probability']:.2f}<br>"
                tooltip += f"<b>Confidence:</b> {confidence:.2f}<br>"
                
            # Add AP density if available
            if 'bssid_count' in row:
                tooltip += f"<b>Access Points:</b> {row['bssid_count']}<br>"
                
            # Close the div
            tooltip += "</div>"
            
            # Add circle
            folium.Circle(
                location=(row['lat_grid'], row['lon_grid']),
                radius=30,  # Radius in meters
                color=html_color,
                fill=True,
                fill_color=html_color,
                fill_opacity=row['opacity'],
                tooltip=tooltip
            ).add_to(coverage_layer)
        
        coverage_layer.add_to(m)
        
        # Add low coverage layer
        low_coverage = grid_stats[grid_stats['rssi_mean'] < -75].copy()
        if len(low_coverage) > 0:
            low_coverage_layer = folium.FeatureGroup(name="Low Coverage Areas")
            
            for _, row in low_coverage.iterrows():
                # Skip rows with missing coordinates
                if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                    continue
                    
                # Create tooltip with detailed information
                tooltip = (
                    f"<div style='font-family: Arial; font-size: 12px;'>"
                    f"<b>RSSI:</b> {row['rssi_mean']:.1f} dBm<br>"
                    f"<b>Sample Count:</b> {row['rssi_count']}<br>"
                    f"<b>Std Dev:</b> {row['rssi_std']:.2f}<br>"
                )
                
                # Add transport mode if available
                if 'predicted_mode' in row:
                    tooltip += f"<b>Transport Mode:</b> {row['predicted_mode']}<br>"
                    
                # Add anomaly info if available
                if 'is_anomaly_mean' in row and row['is_anomaly_mean'] > 0.2:
                    tooltip += f"<b>Anomaly Score:</b> {row['is_anomaly_mean']:.2f} <span style='color:red'>(Warning!)</span><br>"
                    
                # Close the div
                tooltip += "</div>"
                
                # Add circle
                folium.Circle(
                    location=(row['lat_grid'], row['lon_grid']),
                    radius=30,  # Radius in meters
                    color='#d73027',  # Red in color-blind friendly palette
                    fill=True,
                    fill_color='#d73027',
                    fill_opacity=0.7,
                    tooltip=tooltip
                ).add_to(low_coverage_layer)
            
            low_coverage_layer.add_to(m)
        
        # Add sample density heatmap
        heat_data = [[row['lat_grid'], row['lon_grid'], row['rssi_count']] 
                    for _, row in grid_stats.iterrows() 
                    if not pd.isna(row['lat_grid']) and not pd.isna(row['lon_grid'])]
        
        if heat_data:
            # Use colorblind-friendly gradient
            HeatMap(
                heat_data,
                name="Sample Density Heatmap",
                radius=15,
                blur=10,
                gradient={0.2: '#313695', 0.4: '#4575b4', 0.6: '#91bfdb', 0.8: '#fee090', 1.0: '#fc8d59'}
            ).add_to(m)
        
        # Add anomalies if available
        if 'is_anomaly_mean' in grid_stats.columns:
            anomalies = grid_stats[grid_stats['is_anomaly_mean'] > 0.5].copy()
            if len(anomalies) > 0:
                anomaly_layer = folium.FeatureGroup(name="Signal Anomalies")
                
                for _, row in anomalies.iterrows():
                    # Skip rows with missing coordinates
                    if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                        continue
                        
                    # Create tooltip with detailed information
                    tooltip = (
                        f"<div style='font-family: Arial; font-size: 12px;'>"
                        f"<b>RSSI:</b> {row['rssi_mean']:.1f} dBm<br>"
                        f"<b>Anomaly Score:</b> {row['is_anomaly_mean']:.2f}<br>"
                        f"<b>Sample Count:</b> {row['rssi_count']}<br>"
                    )
                    
                    # Add transport mode if available
                    if 'predicted_mode' in row:
                        tooltip += f"<b>Transport Mode:</b> {row['predicted_mode']}<br>"
                    
                    # Try to classify anomaly type based on data patterns
                    if 'rssi_std' in row and 'rssi_change_std' in row:
                        if row['rssi_std'] > 10 and row['rssi_change_std'] > 8:
                            anomaly_type = "High Volatility"
                        elif row['rssi_mean'] < -80 and row['rssi_std'] < 5:
                            anomaly_type = "Persistent Low Signal"
                        elif 'speed_mps' in row and row['speed_mps'] > 5:
                            anomaly_type = "High-Speed Signal Drop"
                        else:
                            anomaly_type = "Unknown Pattern"
                            
                        tooltip += f"<b>Anomaly Type:</b> {anomaly_type}<br>"
                    
                    # Close the div
                    tooltip += "</div>"
                    
                    # Add circle with a distinctive purple color (colorblind visible)
                    folium.Circle(
                        location=(row['lat_grid'], row['lon_grid']),
                        radius=30,  # Radius in meters
                        color='#762a83',  # Purple in colorblind-friendly palette
                        fill=True,
                        fill_color='#762a83',
                        fill_opacity=0.7,
                        tooltip=tooltip
                    ).add_to(anomaly_layer)
                
                anomaly_layer.add_to(m)
        
        # Add legend for color interpretation
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 220px; height: 120px; 
                    border:2px solid grey; z-index:9999; background-color:white;
                    padding: 10px; font-size: 14px; font-family: Arial;">
            <div style="font-weight: bold; margin-bottom: 5px;">RSSI Legend:</div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #fcde9c; width: 20px; height: 15px; margin-right: 5px;"></div>
                <span>Good (> -65 dBm)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #7a88b4; width: 20px; height: 15px; margin-right: 5px;"></div>
                <span>Fair (-75 to -65 dBm)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: #32374a; width: 20px; height: 15px; margin-right: 5px;"></div>
                <span>Poor (< -75 dBm)</span>
            </div>
            <div style="font-style: italic; font-size: 12px; margin-top: 5px;">
                Opacity indicates sample count
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; 
                    z-index:9999; background-color:rgba(255, 255, 255, 0.8);
                    padding: 10px; font-family: Arial; border-radius: 5px;">
            <h3 style="margin: 0;">WiFi Coverage Analysis</h3>
            <div style="font-size: 12px; margin-top: 5px;">
                Showing signal strength with sample density overlay
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map to HTML
        m.save(f"{self.plots_dir}/enhanced_maps/coverage_with_density.html")
        
        return m
    
    def create_transport_mode_filtered_maps(self, grid_stats, merged_df=None, center=None, zoom_start=13):
        """
        Create maps filtered by transport mode
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with coverage and location data
        merged_df : DataFrame, optional
            Merged data with transport mode predictions
        center : tuple, optional
            (latitude, longitude) center of the map
        zoom_start : int, default=13
            Initial zoom level
            
        Returns:
        --------
        maps : dict
            Dictionary of maps by transport mode
        """
        # Check if we have transport mode data
        if 'predicted_mode' not in grid_stats.columns:
            if merged_df is None or 'predicted_mode' not in merged_df.columns:
                print("No transport mode data available. Skipping mode-filtered maps.")
                return None
            
            # Merge transport mode data from merged_df
            print("Merging transport mode data from merged dataframe...")
            
            # First, aggregate transport modes by grid cell
            mode_by_grid = merged_df.groupby(['lat_grid', 'lon_grid'])['predicted_mode'].agg(
                lambda x: x.value_counts().index[0]  # Most common mode
            ).reset_index()
            
            # Merge back to grid_stats
            grid_stats = pd.merge(
                grid_stats,
                mode_by_grid,
                on=['lat_grid', 'lon_grid'],
                how='left'
            )
        
        # Get unique transport modes
        modes = grid_stats['predicted_mode'].dropna().unique()
        
        # Create a map for each mode
        maps = {}
        
        for mode in modes:
            print(f"Creating coverage map for transport mode: {mode}")
            
            # Filter grid_stats for this mode
            mode_stats = grid_stats[grid_stats['predicted_mode'] == mode].copy()
            
            if len(mode_stats) == 0:
                print(f"No data for transport mode: {mode}")
                continue
            
            # Compute center if not provided
            if center is None:
                center = (
                    mode_stats['lat_grid'].mean(),
                    mode_stats['lon_grid'].mean()
                )
            
            # Create base map
            m = folium.Map(
                location=center,
                zoom_start=zoom_start,
                tiles='OpenStreetMap'
            )
            
            # Add different tile layers
            folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
            folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
            
            # Prepare data for visualization
            # Normalize RSSI values for coloring (-85 to -65 dBm)
            mode_stats['rssi_norm'] = np.clip(
                (mode_stats['rssi_mean'] - (-85)) / ((-65) - (-85)),
                0, 1
            )
            
            # Add coverage layer
            coverage_layer = folium.FeatureGroup(name=f"Coverage for {mode}")
            
            for _, row in mode_stats.iterrows():
                # Skip rows with missing coordinates
                if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                    continue
                    
                # Get color based on normalized RSSI
                color = self.coverage_cmap(1 - row['rssi_norm'])  # Invert for red=low, green=high
                html_color = f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.7)"
                
                # Create tooltip with detailed information
                tooltip = (
                    f"RSSI: {row['rssi_mean']:.1f} dBm<br>"
                    f"Sample Count: {row['rssi_count']}<br>"
                    f"Std Dev: {row['rssi_std']:.2f}"
                )
                
                # Add circle
                folium.Circle(
                    location=(row['lat_grid'], row['lon_grid']),
                    radius=30,  # Radius in meters
                    color=html_color,
                    fill=True,
                    fill_color=html_color,
                    fill_opacity=0.7,
                    tooltip=tooltip
                ).add_to(coverage_layer)
            
            coverage_layer.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map to HTML
            m.save(f"{self.plots_dir}/enhanced_maps/coverage_{mode}.html")
            
            # Store map
            maps[mode] = m
        
        return maps
    
    def create_confidence_map(self, grid_stats, probabilities=None, center=None, zoom_start=13):
        """
        Create a map showing coverage with confidence levels
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with coverage data
        probabilities : DataFrame, optional
            Prediction probabilities if not in grid_stats
        center : tuple, optional
            (latitude, longitude) center of the map
        zoom_start : int, default=13
            Initial zoom level
            
        Returns:
        --------
        m : folium.Map
            Folium map object
        """
        print("Creating coverage map with confidence levels...")
        
        # Check if we have probability data
        prob_col = 'coverage_probability'
        if prob_col not in grid_stats.columns:
            if probabilities is not None:
                # Merge probabilities into grid_stats
                grid_stats = pd.merge(
                    grid_stats,
                    probabilities,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            else:
                print(f"No {prob_col} column available. Using binary predictions only.")
                # Create a dummy probability column
                grid_stats[prob_col] = grid_stats['low_coverage_area'].astype(float)
        
        # Compute center if not provided
        if center is None:
            center = (
                grid_stats['lat_grid'].mean(),
                grid_stats['lon_grid'].mean()
            )
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Create a layer for confidence visualization
        confidence_layer = folium.FeatureGroup(name="Coverage with Confidence")
        
        # Normalize confidence for coloring (0.5 to 1.0)
        grid_stats['confidence'] = np.abs(grid_stats[prob_col] - 0.5) * 2  # 0 to 1 scale
        
        for _, row in grid_stats.iterrows():
            # Skip rows with missing coordinates
            if pd.isna(row['lat_grid']) or pd.isna(row['lon_grid']):
                continue
                
            # Determine color based on prediction (green for good coverage, red for low)
            base_color = "red" if row[prob_col] > 0.5 else "green"
            
            # Create tooltip with detailed information
            tooltip = (
                f"RSSI: {row['rssi_mean']:.1f} dBm<br>"
                f"Coverage Probability: {row[prob_col]:.2f}<br>"
                f"Confidence: {row['confidence']:.2f}<br>"
                f"Sample Count: {row['rssi_count']}"
            )
            
            # Use opacity based on confidence
            opacity = 0.3 + (row['confidence'] * 0.6)  # Scale to 0.3-0.9
            
            # Add circle
            folium.Circle(
                location=(row['lat_grid'], row['lon_grid']),
                radius=30,  # Radius in meters
                color=base_color,
                fill=True,
                fill_color=base_color,
                fill_opacity=opacity,
                tooltip=tooltip
            ).add_to(confidence_layer)
        
        confidence_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map to HTML
        m.save(f"{self.plots_dir}/enhanced_maps/coverage_with_confidence.html")
        
        return m
    
    def create_static_density_plot(self, grid_stats, low_threshold=-75):
        """
        Create a static plot showing coverage with sample density
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with coverage data
        low_threshold : float, default=-75
            RSSI threshold for low coverage
        """
        print("Creating static coverage plot with sample density...")
        
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot with size representing sample density
        # and color representing RSSI
        scatter = plt.scatter(
            grid_stats['lon_grid'], 
            grid_stats['lat_grid'],
            c=grid_stats['rssi_mean'],
            s=grid_stats['rssi_count'] / 5,  # Scale down for visibility
            cmap=self.coverage_cmap,
            alpha=0.7,
            edgecolors='k',
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('RSSI (dBm)')
        
        # Add a marker for low coverage areas
        low_coverage = grid_stats[grid_stats['rssi_mean'] < low_threshold]
        if len(low_coverage) > 0:
            plt.scatter(
                low_coverage['lon_grid'], 
                low_coverage['lat_grid'],
                marker='x',
                color='black',
                s=50,
                alpha=0.7,
                label=f'Low Coverage (< {low_threshold} dBm)'
            )
        
        # Add legend for bubble size
        sizes = [10, 50, 100, 200]
        labels = []
        
        # Calculate what rssi_count values correspond to these sizes
        max_count = grid_stats['rssi_count'].max()
        for size in sizes:
            count = size * 5  # Reverse the scaling
            if count <= max_count:
                labels.append(f"{count} samples")
                
        if labels:
            # Create legend elements
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=np.sqrt(size/np.pi), label=label)
                for size, label in zip(sizes, labels)
            ]
            
            # Add low coverage legend if applicable
            if len(low_coverage) > 0:
                legend_elements.append(
                    Line2D([0], [0], marker='x', color='black', 
                          markersize=10, label=f'Low Coverage (< {low_threshold} dBm)')
                )
                
            plt.legend(handles=legend_elements, title="Sample Density", loc='upper right')
        
        # Add titles and labels
        plt.title('WiFi Coverage with Sample Density', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/enhanced_maps/static_coverage_with_density.png", dpi=300)
        plt.close()
    
    def create_comparison_maps(self, grid_stats, field_values=None, transport_modes=None):
        """
        Create comparison maps for different field values or transport modes
        
        Parameters:
        -----------
        grid_stats : DataFrame
            Grid statistics with coverage data
        field_values : list, optional
            List of field values to compare
        transport_modes : list, optional
            List of transport modes to compare
            
        Returns:
        --------
        maps : dict
            Dictionary of maps by field or mode
        """
        maps = {}
        
        # Create field comparison maps if specified
        if field_values is not None and 'field_id' in grid_stats.columns:
            for field in field_values:
                field_stats = grid_stats[grid_stats['field_id'] == field].copy()
                
                if len(field_stats) == 0:
                    print(f"No data for field: {field}")
                    continue
                
                print(f"Creating comparison map for field: {field}")
                maps[f"field_{field}"] = self.create_density_overlay_map(
                    field_stats,
                    center=None,
                    zoom_start=14
                )
        
        # Create transport mode comparison maps if specified
        if transport_modes is not None and 'predicted_mode' in grid_stats.columns:
            for mode in transport_modes:
                mode_stats = grid_stats[grid_stats['predicted_mode'] == mode].copy()
                
                if len(mode_stats) == 0:
                    print(f"No data for transport mode: {mode}")
                    continue
                
                print(f"Creating comparison map for transport mode: {mode}")
                maps[f"mode_{mode}"] = self.create_density_overlay_map(
                    mode_stats,
                    center=None,
                    zoom_start=14
                )
        
        return maps

def enhance_visualizations(grid_stats, merged_df=None, output_dir='output', plots_dir='plots'):
    """
    Create enhanced visualizations for the WiFi coverage data
    
    Parameters:
    -----------
    grid_stats : DataFrame
        Grid statistics with coverage data
    merged_df : DataFrame, optional
        Merged data with transport mode predictions
    output_dir : str, default='output'
        Directory to save output files
    plots_dir : str, default='plots'
        Directory to save plots
        
    Returns:
    --------
    visualizer : EnhancedCoverageVisualizer
        Visualizer object with created maps
    """
    print("Creating enhanced visualizations...")
    
    # Create visualizer
    visualizer = EnhancedCoverageVisualizer(
        output_dir=output_dir,
        plots_dir=plots_dir
    )
    
    # Create density overlay map
    density_map = visualizer.create_density_overlay_map(grid_stats)
    
    # Create confidence map if probability column is available
    if 'coverage_probability' in grid_stats.columns:
        confidence_map = visualizer.create_confidence_map(grid_stats)
    
    # Create transport mode filtered maps if available
    if 'predicted_mode' in grid_stats.columns or (merged_df is not None and 'predicted_mode' in merged_df.columns):
        mode_maps = visualizer.create_transport_mode_filtered_maps(grid_stats, merged_df)
    
    # Create static density plot
    visualizer.create_static_density_plot(grid_stats)
    
    return visualizer

if __name__ == "__main__":
    # Test the module with sample data
    try:
        print("Loading grid statistics...")
        try:
            grid_stats = pd.read_csv('grid_coverage_statistics.csv')
            
            # Try to load merged data with transport modes if available
            try:
                merged_df = pd.read_csv('transport_modes.csv')
                print(f"Loaded transport modes with {len(merged_df)} rows")
            except FileNotFoundError:
                print("Transport modes not found. Running visualizations without mode stratification.")
                merged_df = None
            
            # Create enhanced visualizations
            visualizer = enhance_visualizations(grid_stats, merged_df)
            
        except FileNotFoundError:
            print("Grid statistics not found. Please run data preprocessing first.")
        
    except Exception as e:
        print(f"Error during visualization: {e}") 