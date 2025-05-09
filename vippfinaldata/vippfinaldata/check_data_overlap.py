import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load first 10000 rows of each dataset
print("Loading data samples...")
wifi_df = pd.read_csv('Hips_WiFi.csv', nrows=10000)
location_df = pd.read_csv('Hips_Location.csv', nrows=10000)

# Convert timestamps to datetime
wifi_df['timestamp_dt'] = pd.to_datetime(wifi_df['timestamp'], unit='ms')
location_df['timestamp_dt'] = pd.to_datetime(location_df['timestamp_ms'], unit='ms')

# Print timestamp ranges
print("\nTimestamp Ranges:")
print(f"WiFi: {wifi_df['timestamp_dt'].min()} to {wifi_df['timestamp_dt'].max()}")
print(f"Location: {location_df['timestamp_dt'].min()} to {location_df['timestamp_dt'].max()}")

# Add timestamp_ms to WiFi for comparison
wifi_df['timestamp_ms'] = wifi_df['timestamp']

# Check for overlapping time ranges
min_wifi = wifi_df['timestamp_ms'].min()
max_wifi = wifi_df['timestamp_ms'].max()
min_loc = location_df['timestamp_ms'].min()
max_loc = location_df['timestamp_ms'].max()

# Calculate overlap
has_overlap = not (max_wifi < min_loc or min_wifi > max_loc)
overlap_start = max(min_wifi, min_loc)
overlap_end = min(max_wifi, max_loc)

print(f"\nTimestamp Overlap: {has_overlap}")
if has_overlap:
    overlap_pct_wifi = (min(max_wifi, max_loc) - max(min_wifi, min_loc)) / (max_wifi - min_wifi) * 100
    overlap_pct_loc = (min(max_wifi, max_loc) - max(min_wifi, min_loc)) / (max_loc - min_loc) * 100
    print(f"Overlap period: {pd.to_datetime(overlap_start, unit='ms')} to {pd.to_datetime(overlap_end, unit='ms')}")
    print(f"Overlap percentage of WiFi data: {overlap_pct_wifi:.2f}%")
    print(f"Overlap percentage of Location data: {overlap_pct_loc:.2f}%")

# Try a merge on the timestamp_ms with a tolerance
print("\nTesting merge with various tolerances...")
for tolerance_sec in [1, 5, 10, 30, 60, 300]:
    try:
        merged = pd.merge_asof(
            wifi_df.sort_values('timestamp_ms'),
            location_df.sort_values('timestamp_ms'),
            on='timestamp_ms',
            direction='nearest',
            tolerance=tolerance_sec * 1000  # Convert to milliseconds
        )
        join_rate = merged['latitude_deg'].notna().mean()
        print(f"Tolerance {tolerance_sec}s: Join rate = {join_rate:.2%}")
    except Exception as e:
        print(f"Tolerance {tolerance_sec}s: Error - {e}")

# Try using a window to find closest location record for each WiFi record
print("\nAnalyzing closest timestamps...")
closest_diffs = []
for wifi_ts in wifi_df['timestamp_ms'].head(100):
    diffs = abs(location_df['timestamp_ms'] - wifi_ts)
    min_diff = diffs.min()
    closest_diffs.append(min_diff / 1000)  # Convert to seconds

closest_diffs = np.array(closest_diffs)
print(f"Minimum time difference (sec): {closest_diffs.min():.2f}")
print(f"Maximum time difference (sec): {closest_diffs.max():.2f}")
print(f"Mean time difference (sec): {closest_diffs.mean():.2f}")
print(f"Median time difference (sec): {np.median(closest_diffs):.2f}")

# Suggest improvements
print("\nSuggested fixes:")
if not has_overlap:
    print("- The datasets have no overlapping timestamps. Try using larger datasets or check data collection periods.")
elif closest_diffs.min() > 5:
    print(f"- Increase merge tolerance to at least {max(60, int(np.median(closest_diffs) * 1.5))} seconds.")
    print("- Consider using regular merge on timestamp_ms with a suitable time window instead of merge_asof.")
else:
    print(f"- Use a tolerance of at least {max(5, int(np.median(closest_diffs) * 1.5))} seconds with merge_asof.")
    
print("\nData sample (first 5 rows):")
print("\nWiFi data:")
print(wifi_df[['timestamp', 'timestamp_dt', 'bssid', 'rssi']].head())
print("\nLocation data:")
print(location_df[['timestamp_ms', 'timestamp_dt', 'latitude_deg', 'longitude_deg']].head()) 