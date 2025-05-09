import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("Analyzing dataset time windows to find optimal overlap...")

# Load all timestamps from both datasets
print("Loading WiFi timestamps...")
wifi_timestamps = pd.read_csv('Hips_WiFi.csv', usecols=['timestamp'])['timestamp']
wifi_timestamps = pd.to_datetime(wifi_timestamps, unit='ms')

print("Loading Location timestamps...")
loc_timestamps = pd.read_csv('Hips_Location.csv', usecols=['timestamp_ms'])['timestamp_ms']
loc_timestamps = pd.to_datetime(loc_timestamps, unit='ms')

# Get time ranges
print("\nOverall Time Ranges:")
wifi_start, wifi_end = wifi_timestamps.min(), wifi_timestamps.max()
loc_start, loc_end = loc_timestamps.min(), loc_timestamps.max()
print(f"WiFi data: {wifi_start} to {wifi_end} ({len(wifi_timestamps):,} records)")
print(f"Location data: {loc_start} to {loc_end} ({len(loc_timestamps):,} records)")

# Create time bins (1-hour intervals) and count records in each
print("\nAnalyzing hourly distribution...")
interval = timedelta(hours=1)
bins = pd.date_range(start=min(wifi_start, loc_start), end=max(wifi_end, loc_end), freq=interval)

wifi_counts = pd.cut(wifi_timestamps, bins=bins).value_counts().sort_index()
loc_counts = pd.cut(loc_timestamps, bins=bins).value_counts().sort_index()

# Create a DataFrame for analysis
hourly_df = pd.DataFrame({
    'hour': bins[:-1],
    'wifi_count': wifi_counts.values,
    'loc_count': loc_counts.values
})

# Calculate overlap scores (min of both counts)
hourly_df['overlap'] = hourly_df.apply(lambda x: min(x['wifi_count'], x['loc_count']), axis=1)
hourly_df['overlap_pct'] = hourly_df['overlap'] / hourly_df.apply(lambda x: max(x['wifi_count'], x['loc_count']), axis=1) * 100

# Find best windows based on overlap
window_sizes = [1, 2, 3, 4]  # hours
best_windows = {}

for window_size in window_sizes:
    # Calculate rolling sum for each window size
    hourly_df[f'wifi_rolling_{window_size}h'] = hourly_df['wifi_count'].rolling(window_size).sum()
    hourly_df[f'loc_rolling_{window_size}h'] = hourly_df['loc_count'].rolling(window_size).sum()
    hourly_df[f'overlap_rolling_{window_size}h'] = hourly_df[[f'wifi_rolling_{window_size}h', f'loc_rolling_{window_size}h']].min(axis=1)
    
    # Find window with max overlap
    best_idx = hourly_df[f'overlap_rolling_{window_size}h'].idxmax()
    if pd.isna(best_idx):
        continue
        
    start_time = hourly_df.loc[best_idx - window_size + 1, 'hour'] if best_idx >= window_size - 1 else hourly_df.loc[0, 'hour']
    end_time = hourly_df.loc[best_idx, 'hour'] + interval
    
    wifi_in_window = hourly_df.loc[best_idx - window_size + 1:best_idx, 'wifi_count'].sum() if best_idx >= window_size - 1 else hourly_df.loc[:best_idx, 'wifi_count'].sum()
    loc_in_window = hourly_df.loc[best_idx - window_size + 1:best_idx, 'loc_count'].sum() if best_idx >= window_size - 1 else hourly_df.loc[:best_idx, 'loc_count'].sum()
    
    best_windows[window_size] = {
        'start': start_time,
        'end': end_time,
        'wifi_count': int(wifi_in_window),
        'loc_count': int(loc_in_window),
        'overlap': min(wifi_in_window, loc_in_window),
        'wifi_timestamp_start': int(start_time.timestamp() * 1000),
        'wifi_timestamp_end': int(end_time.timestamp() * 1000)
    }

# Print best windows
print("\nBest time windows for data overlap:")
for window_size, window_info in best_windows.items():
    print(f"\n{window_size}-hour window:")
    print(f"  Time range: {window_info['start']} to {window_info['end']}")
    print(f"  WiFi records: {window_info['wifi_count']:,}")
    print(f"  Location records: {window_info['loc_count']:,}")
    print(f"  Timestamp filter (ms): {window_info['wifi_timestamp_start']} to {window_info['wifi_timestamp_end']}")
    
# Plot the hourly distribution
plt.figure(figsize=(12, 6))
plt.bar(hourly_df['hour'], hourly_df['wifi_count'], alpha=0.5, label='WiFi')
plt.bar(hourly_df['hour'], hourly_df['loc_count'], alpha=0.5, label='Location')
plt.xlabel('Time')
plt.ylabel('Record Count')
plt.title('Hourly Distribution of WiFi and Location Records')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data_distribution.png')
print("\nSaved hourly distribution plot to 'data_distribution.png'")

# Recommend the best window
best_window_size = max(best_windows.items(), key=lambda x: x[1]['overlap'])[0]
best_window = best_windows[best_window_size]

print("\nRecommended data window:")
print(f"Use a {best_window_size}-hour window from {best_window['start']} to {best_window['end']}")
print(f"WiFi timestamp filter: {best_window['wifi_timestamp_start']} to {best_window['wifi_timestamp_end']}")
print("\nExample command:")
print(f"py -3 run_wifi_pipeline.py --wifi Hips_WiFi.csv --location Hips_Location.csv --threshold -75 --wifi-start {best_window['wifi_timestamp_start']} --wifi-end {best_window['wifi_timestamp_end']}") 