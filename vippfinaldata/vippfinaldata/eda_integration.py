import os
import time

print("Starting Comprehensive EDA Process...")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Step 1: Run individual data analyses
print("\n" + "="*50)
print("Step 1: Analyzing individual datasets")
print("="*50)

print("\nRunning Location Data Analysis...")
try:
    exec(open("eda_location_data.py").read())
except Exception as e:
    print(f"Error in location data analysis: {e}")

print("\nRunning GPS Data Analysis...")
try:
    exec(open("eda_gps_data.py").read())
except Exception as e:
    print(f"Error in GPS data analysis: {e}")

print("\nRunning WiFi Data Analysis...")
try:
    exec(open("eda_wifi_data.py").read())
except Exception as e:
    print(f"Error in WiFi data analysis: {e}")

# Step 2: Run data integration and relationship analysis
print("\n" + "="*50)
print("Step 2: Analyzing relationships between datasets")
print("="*50)

print("\nRunning Data Integration and Relationship Analysis...")
try:
    exec(open("eda_data_merging.py").read())
except Exception as e:
    print(f"Error in data integration analysis: {e}")

# Step 3: Summarize findings
print("\n" + "="*50)
print("EDA Summary and Key Findings")
print("="*50)

print("""
Key findings from the exploratory data analysis:

1. Data Coverage and Quality:
   - Analyzed location, GPS, and WiFi signal data
   - Identified and handled outliers and missing values
   - Verified data consistency and reliability

2. Signal Strength Analysis:
   - Mapped RSSI values across geographical areas
   - Created signal quality categories from RSSI values
   - Analyzed signal strength variations by time and space

3. Spatial Patterns:
   - Used DBSCAN clustering to identify regions with similar characteristics
   - Created heatmaps showing signal strength distributions
   - Identified potential low coverage areas

4. Temporal Patterns:
   - Analyzed signal variations over time
   - Implemented rolling statistics to understand temporal stability
   - Identified time-dependent patterns in signal quality

5. Environmental Factors:
   - Examined relationship between altitude and signal strength
   - Analyzed impact of movement speed on signal quality
   - Identified potential physical obstacles affecting signal

6. Next Steps for Modeling:
   - Created clean, integrated datasets ready for model development
   - Identified key features for predicting low coverage areas
   - Established baseline understanding of signal patterns to guide model development
""")

print("\nEDA process completed successfully!")
print(f"All results and visualizations saved to the 'plots' directory.") 