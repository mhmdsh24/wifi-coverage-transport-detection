import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from eda_utils import *

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("Starting WiFi Data Analysis...")

# Load WiFi data with sampling due to large size
# Using 50,000 rows for analysis to handle memory constraints
wifi_df = load_data('Hips_WiFi.csv', sample_size=50000)

# Basic analysis
analyze_dataframe(wifi_df, "WiFi")

# Handle any missing values
wifi_df = handle_missing_values(wifi_df)

# Specific analysis for WiFi data
if wifi_df is not None:
    # RSSI value analysis
    print("\nRSSI Value Analysis:")
    print(f"Min RSSI: {wifi_df['rssi'].min()}")
    print(f"Max RSSI: {wifi_df['rssi'].max()}")
    print(f"Mean RSSI: {wifi_df['rssi'].mean():.2f}")
    print(f"Median RSSI: {wifi_df['rssi'].median()}")
    
    # Plot RSSI distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(wifi_df['rssi'], kde=True, bins=30)
    plt.title('Distribution of RSSI Values')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/wifi_rssi_distribution.png")
    plt.close()
    
    # Create bins for signal strength categories
    wifi_df['signal_quality'] = pd.cut(
        wifi_df['rssi'],
        bins=[-100, -80, -70, -60, -50, 0],
        labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
    )
    
    # Plot signal quality categories
    plt.figure(figsize=(12, 6))
    signal_counts = wifi_df['signal_quality'].value_counts().sort_index()
    signal_counts.plot(kind='bar', color=sns.color_palette("RdYlGn", n_colors=len(signal_counts)))
    plt.title('WiFi Signal Quality Categories')
    plt.xlabel('Signal Quality')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/wifi_signal_quality.png")
    plt.close()
    
    # Analyze unique BSSIDs and SSIDs
    unique_bssids = wifi_df['bssid'].nunique()
    unique_ssids = wifi_df['ssid'].nunique()
    print(f"\nUnique BSSIDs: {unique_bssids}")
    print(f"Unique SSIDs: {unique_ssids}")
    
    # Top SSIDs by frequency
    print("\nTop 10 most frequent SSIDs:")
    print(wifi_df['ssid'].value_counts().head(10))
    
    # RSSI by frequency band
    wifi_df['frequency_band'] = np.where(wifi_df['freq'] > 3000, '5GHz', '2.4GHz')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='frequency_band', y='rssi', data=wifi_df)
    plt.title('RSSI by Frequency Band')
    plt.xlabel('Frequency Band')
    plt.ylabel('RSSI (dBm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/wifi_rssi_by_frequency_band.png")
    plt.close()
    
    # Time-based analysis
    wifi_df['timestamp_ms'] = pd.to_numeric(wifi_df['timestamp'], errors='coerce')
    wifi_df['timestamp_dt'] = pd.to_datetime(wifi_df['timestamp_ms'], unit='ms', errors='coerce')
    
    if not wifi_df['timestamp_dt'].isnull().all():
        wifi_df['hour'] = wifi_df['timestamp_dt'].dt.hour
        
        plt.figure(figsize=(12, 6))
        hourly_rssi = wifi_df.groupby('hour')['rssi'].mean().reset_index()
        sns.lineplot(x='hour', y='rssi', data=hourly_rssi, marker='o')
        plt.title('Average RSSI by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average RSSI (dBm)')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/wifi_rssi_by_hour.png")
        plt.close()
    
    # Security analysis
    wifi_df['has_wpa2'] = wifi_df['caps'].str.contains('WPA2', case=False, na=False)
    wifi_df['has_wpa'] = wifi_df['caps'].str.contains('WPA(?!2)', case=False, na=False, regex=True)
    wifi_df['has_wps'] = wifi_df['caps'].str.contains('WPS', case=False, na=False)
    
    # Count security types
    security_counts = {
        'WPA2': wifi_df['has_wpa2'].sum(),
        'WPA': wifi_df['has_wpa'].sum(),
        'WPS': wifi_df['has_wps'].sum()
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(security_counts.keys(), security_counts.values())
    plt.title('WiFi Security Types')
    plt.xlabel('Security Type')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/wifi_security_types.png")
    plt.close()
    
    # RSSI by security type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='has_wpa2', y='rssi', data=wifi_df)
    plt.title('RSSI by WPA2 Security')
    plt.xlabel('Has WPA2')
    plt.ylabel('RSSI (dBm)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/wifi_rssi_by_wpa2.png")
    plt.close()
    
    # Signal quality by access point - top 10 most frequent BSSIDs
    top_bssids = wifi_df['bssid'].value_counts().head(10).index
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='bssid', y='rssi', data=wifi_df[wifi_df['bssid'].isin(top_bssids)])
    plt.title('RSSI by Top 10 Access Points')
    plt.xlabel('BSSID')
    plt.ylabel('RSSI (dBm)')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/wifi_rssi_by_top_bssids.png")
    plt.close()
    
    # Calculate rolling statistics for RSSI
    # First sort by timestamp to ensure proper sequence
    if not wifi_df['timestamp_ms'].isnull().all():
        # Group by BSSID to analyze each access point separately
        # Use the 10 most frequent BSSIDs for this analysis
        for bssid in top_bssids[:3]:  # Limiting to 3 for brevity
            bssid_df = wifi_df[wifi_df['bssid'] == bssid].sort_values('timestamp_ms')
            
            if len(bssid_df) > 10:  # Need enough data points for rolling stats
                # Calculate rolling statistics with a window of 10
                bssid_df['rssi_rolling_mean'] = bssid_df['rssi'].rolling(window=10, min_periods=1).mean()
                bssid_df['rssi_rolling_std'] = bssid_df['rssi'].rolling(window=10, min_periods=1).std()
                
                # Plot the rolling statistics
                plt.figure(figsize=(14, 8))
                plt.plot(bssid_df['timestamp_ms'], bssid_df['rssi'], label='RSSI', alpha=0.5)
                plt.plot(bssid_df['timestamp_ms'], bssid_df['rssi_rolling_mean'], label='Rolling Mean', linewidth=2)
                plt.fill_between(
                    bssid_df['timestamp_ms'],
                    bssid_df['rssi_rolling_mean'] - bssid_df['rssi_rolling_std'],
                    bssid_df['rssi_rolling_mean'] + bssid_df['rssi_rolling_std'],
                    alpha=0.2, label='±1 Std Dev'
                )
                plt.title(f'Rolling Statistics for BSSID: {bssid}')
                plt.xlabel('Timestamp')
                plt.ylabel('RSSI (dBm)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"plots/wifi_rolling_stats_{bssid.replace(':', '_')}.png")
                plt.close()
    
    # Save cleaned dataframe
    wifi_df.to_csv('cleaned_wifi_data.csv', index=False)
    print("\nCleaned WiFi data saved to cleaned_wifi_data.csv")
    
print("WiFi Data Analysis completed!") 