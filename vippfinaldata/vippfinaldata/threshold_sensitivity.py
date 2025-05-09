import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score
import os
from matplotlib.colors import LinearSegmentedColormap

class ThresholdSensitivityAnalyzer:
    """
    Threshold Sensitivity Analysis for WiFi Coverage
    
    This class analyzes how different RSSI thresholds impact coverage predictions,
    allowing users to make data-driven decisions about optimal thresholds.
    """
    
    def __init__(self, output_dir='output', plots_dir='plots'):
        """
        Initialize the analyzer
        
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
        os.makedirs(f"{plots_dir}/threshold_analysis", exist_ok=True)
        
        # Store results
        self.results = {}
        
        # Define colorblind-safe colormaps
        self.cmap_viridis = plt.cm.viridis
        self.cmap_cividis = plt.cm.cividis  # More colorblind-friendly
        
        # Custom colormap for thresholds (green-yellow-red)
        self.coverage_cmap = LinearSegmentedColormap.from_list(
            'coverage_cmap', ['#1a9850', '#ffffbf', '#d73027']
        )
    
    def analyze_thresholds(self, df, rssi_column='rssi', threshold_range=None, field_value=None, 
                           transport_mode=None, three_class=False):
        """
        Analyze different RSSI thresholds
        
        Parameters:
        -----------
        df : DataFrame
            Data with RSSI values
        rssi_column : str
            Column name for RSSI values
        threshold_range : tuple or list
            Range of thresholds to analyze (min, max, step)
            Default: (-85, -65, 1)
        field_value : str, optional
            If provided, limit analysis to a specific field value
        transport_mode : str, optional
            If provided, limit analysis to a specific transport mode
        three_class : bool, default=False
            If True, analyze thresholds for three classes (good/fair/poor)
            
        Returns:
        --------
        results : dict
            Threshold analysis results
        """
        print("Analyzing RSSI threshold sensitivity...")
        
        if threshold_range is None:
            threshold_range = (-85, -65, 1)
        
        # Extract threshold values
        thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_range[2], threshold_range[2])
        
        # Filter data if field_value is specified
        if field_value is not None and 'field_id' in df.columns:
            df = df[df['field_id'] == field_value].copy()
            print(f"Filtered to field: {field_value} ({len(df)} rows)")
        
        # Filter by transport mode if specified
        if transport_mode is not None and 'predicted_mode' in df.columns:
            df = df[df['predicted_mode'] == transport_mode].copy()
            print(f"Filtered to transport mode: {transport_mode} ({len(df)} rows)")
        
        # Initialize result arrays
        coverage_pct = []
        threshold_values = []
        precision_values = []
        recall_values = []
        f1_values = []
        
        # For three-class analysis
        if three_class:
            good_pct = []
            fair_pct = []
            poor_pct = []
            
            # Define second threshold (for fair/good boundary)
            second_thresholds = np.arange(-70, -60, 1)
        
        # For each threshold, calculate coverage percentage
        for threshold in thresholds:
            if not three_class:
                # Binary classification (low/good coverage)
                df['low_coverage'] = (df[rssi_column] < threshold).astype(int)
                
                # Calculate coverage percentage
                coverage_percent = df['low_coverage'].mean() * 100
                coverage_pct.append(coverage_percent)
                threshold_values.append(threshold)
                
                # If truth values are available, calculate precision, recall, F1
                if 'is_low_coverage_truth' in df.columns:
                    precision_values.append(
                        precision_score(df['is_low_coverage_truth'], df['low_coverage'])
                    )
                    recall_values.append(
                        recall_score(df['is_low_coverage_truth'], df['low_coverage'])
                    )
                    f1_values.append(
                        f1_score(df['is_low_coverage_truth'], df['low_coverage'])
                    )
            else:
                # Three-class analysis results
                second_threshold_results = []
                
                for second_threshold in second_thresholds:
                    if second_threshold <= threshold:
                        continue  # Skip invalid threshold combinations
                        
                    # Create three classes
                    df['coverage_class'] = pd.cut(
                        df[rssi_column],
                        bins=[-100, threshold, second_threshold, 0],
                        labels=['poor', 'fair', 'good']
                    )
                    
                    # Calculate percentages
                    class_counts = df['coverage_class'].value_counts(normalize=True) * 100
                    poor_pct_val = class_counts.get('poor', 0)
                    fair_pct_val = class_counts.get('fair', 0)
                    good_pct_val = class_counts.get('good', 0)
                    
                    second_threshold_results.append({
                        'poor_threshold': threshold,
                        'good_threshold': second_threshold,
                        'poor_pct': poor_pct_val,
                        'fair_pct': fair_pct_val,
                        'good_pct': good_pct_val,
                    })
                
                if second_threshold_results:
                    # Find best balance (closest to 33% each)
                    best_balance = min(second_threshold_results, 
                                     key=lambda x: abs(x['poor_pct'] - 33.3) + 
                                                 abs(x['fair_pct'] - 33.3) + 
                                                 abs(x['good_pct'] - 33.3))
                    
                    threshold_values.append(threshold)
                    coverage_pct.append(best_balance['poor_pct'])
                    good_pct.append(best_balance['good_pct'])
                    fair_pct.append(best_balance['fair_pct'])
                    poor_pct.append(best_balance['poor_pct'])
        
        # Create results dataframe
        if not three_class:
            results_df = pd.DataFrame({
                'threshold': threshold_values,
                'coverage_pct': coverage_pct
            })
            
            # Add precision, recall, F1 if available
            if precision_values:
                results_df['precision'] = precision_values
                results_df['recall'] = recall_values
                results_df['f1_score'] = f1_values
        else:
            results_df = pd.DataFrame({
                'poor_threshold': threshold_values,
                'poor_pct': poor_pct,
                'fair_pct': fair_pct,
                'good_pct': good_pct
            })
        
        # Store results
        analysis_key = f"field_{field_value}_mode_{transport_mode}" if field_value or transport_mode else "all_data"
        if three_class:
            analysis_key += "_three_class"
            
        self.results[analysis_key] = results_df
        
        # Create plots
        self._create_plots(results_df, analysis_key, three_class=three_class)
        
        # Save results to CSV
        results_df.to_csv(f"{self.output_dir}/threshold_analysis_{analysis_key}.csv", index=False)
        
        return results_df
    
    def _create_plots(self, results_df, analysis_key, three_class=False):
        """
        Create plots for threshold analysis
        
        Parameters:
        -----------
        results_df : DataFrame
            Results of threshold analysis
        analysis_key : str
            Key to identify the analysis
        three_class : bool, default=False
            If True, create plots for three-class analysis
        """
        if not three_class:
            # Plot 1: Coverage percentage by threshold
            plt.figure(figsize=(12, 6))
            plt.plot(results_df['threshold'], results_df['coverage_pct'], 'o-', linewidth=2)
            plt.xlabel('RSSI Threshold (dBm)')
            plt.ylabel('Low Coverage Percentage (%)')
            plt.title('Coverage Percentage by RSSI Threshold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/threshold_analysis/coverage_pct_{analysis_key}.png")
            plt.close()
            
            # Plot 2: Precision-Recall if available
            if 'precision' in results_df.columns and 'recall' in results_df.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2, label='Precision')
                plt.plot(results_df['threshold'], results_df['recall'], 'r-', linewidth=2, label='Recall')
                plt.plot(results_df['threshold'], results_df['f1_score'], 'g-', linewidth=2, label='F1 Score')
                plt.xlabel('RSSI Threshold (dBm)')
                plt.ylabel('Score')
                plt.title('Precision, Recall, and F1 Score by RSSI Threshold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.plots_dir}/threshold_analysis/precision_recall_{analysis_key}.png")
                plt.close()
                
                # Plot 3: Precision-Recall curve
                plt.figure(figsize=(10, 10))
                plt.plot(results_df['recall'], results_df['precision'], 'bo-', linewidth=2)
                
                # Add threshold values as text
                for i, threshold in enumerate(results_df['threshold']):
                    plt.annotate(
                        f"{threshold}",
                        (results_df['recall'].iloc[i], results_df['precision'].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve with RSSI Thresholds')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 1.05)
                plt.ylim(0, 1.05)
                plt.tight_layout()
                plt.savefig(f"{self.plots_dir}/threshold_analysis/pr_curve_{analysis_key}.png")
                plt.close()
                
                # Plot 4: ROC curve
                plt.figure(figsize=(10, 10))
                if 'precision' in results_df.columns and 'recall' in results_df.columns:
                    # Calculate FPR (1-precision) and TPR (recall)
                    fpr = 1 - results_df['precision']
                    tpr = results_df['recall']
                    
                    plt.plot(fpr, tpr, 'bo-', linewidth=2)
                    
                    # Add threshold values as text
                    for i, threshold in enumerate(results_df['threshold']):
                        plt.annotate(
                            f"{threshold}",
                            (fpr.iloc[i], tpr.iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center'
                        )
                    
                    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                    plt.xlabel('False Positive Rate (1-Precision)')
                    plt.ylabel('True Positive Rate (Recall)')
                    plt.title('ROC Curve with RSSI Thresholds')
                    plt.grid(True, alpha=0.3)
                    plt.xlim(0, 1.05)
                    plt.ylim(0, 1.05)
                    plt.tight_layout()
                    plt.savefig(f"{self.plots_dir}/threshold_analysis/roc_curve_{analysis_key}.png")
                    plt.close()
        else:
            # Plot for three-class analysis
            plt.figure(figsize=(12, 6))
            
            # Stacked bar chart
            plt.bar(results_df['poor_threshold'], results_df['poor_pct'], label='Poor', color='red', alpha=0.7)
            plt.bar(results_df['poor_threshold'], results_df['fair_pct'], bottom=results_df['poor_pct'], 
                   label='Fair', color='yellow', alpha=0.7)
            plt.bar(results_df['poor_threshold'], results_df['good_pct'], 
                   bottom=results_df['poor_pct'] + results_df['fair_pct'], 
                   label='Good', color='green', alpha=0.7)
            
            plt.xlabel('Poor Coverage Threshold (dBm)')
            plt.ylabel('Percentage (%)')
            plt.title('Three-Class Coverage Distribution by RSSI Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/threshold_analysis/three_class_{analysis_key}.png")
            plt.close()
    
    def analyze_by_transport_mode(self, df, rssi_column='rssi', threshold_range=None):
        """
        Analyze thresholds for each transport mode
        
        Parameters:
        -----------
        df : DataFrame
            Data with RSSI values and transport mode predictions
        rssi_column : str
            Column name for RSSI values
        threshold_range : tuple or list
            Range of thresholds to analyze (min, max, step)
            
        Returns:
        --------
        results : dict
            Threshold analysis results by transport mode
        """
        if 'predicted_mode' not in df.columns:
            print("Transport mode column not found. Run transport mode detection first.")
            return None
        
        # Get unique transport modes
        transport_modes = df['predicted_mode'].unique()
        
        # Analyze for each transport mode
        mode_results = {}
        for mode in transport_modes:
            print(f"\nAnalyzing threshold sensitivity for transport mode: {mode}")
            mode_results[mode] = self.analyze_thresholds(
                df, 
                rssi_column=rssi_column,
                threshold_range=threshold_range,
                transport_mode=mode
            )
        
        # Create comparative plots
        self._create_comparative_plots(mode_results)
        
        return mode_results
    
    def _create_comparative_plots(self, mode_results):
        """
        Create comparative plots for different transport modes
        
        Parameters:
        -----------
        mode_results : dict
            Results for each transport mode
        """
        # Create comparative coverage plot
        plt.figure(figsize=(12, 6))
        
        for mode, results_df in mode_results.items():
            plt.plot(results_df['threshold'], results_df['coverage_pct'], 'o-', linewidth=2, label=f"{mode}")
        
        plt.xlabel('RSSI Threshold (dBm)')
        plt.ylabel('Low Coverage Percentage (%)')
        plt.title('Coverage Percentage by RSSI Threshold and Transport Mode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/threshold_analysis/coverage_pct_by_mode.png")
        plt.close()
        
        # Create comparative F1 plot if available
        if all('f1_score' in df.columns for df in mode_results.values()):
            plt.figure(figsize=(12, 6))
            
            for mode, results_df in mode_results.items():
                plt.plot(results_df['threshold'], results_df['f1_score'], 'o-', linewidth=2, label=f"{mode}")
            
            plt.xlabel('RSSI Threshold (dBm)')
            plt.ylabel('F1 Score')
            plt.title('F1 Score by RSSI Threshold and Transport Mode')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.plots_dir}/threshold_analysis/f1_by_mode.png")
            plt.close()
    
    def recommend_thresholds(self):
        """
        Recommend optimal thresholds based on analysis
        
        Returns:
        --------
        recommendations : dict
            Recommended thresholds for different scenarios
        """
        recommendations = {}
        
        for key, results_df in self.results.items():
            # If we have precision/recall metrics
            if 'f1_score' in results_df.columns:
                # Find threshold with highest F1 score
                best_f1_idx = results_df['f1_score'].idxmax()
                best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
                
                # Find threshold with good precision (> 0.8 if available)
                precision_thresholds = results_df[results_df['precision'] >= 0.8]
                if len(precision_thresholds) > 0:
                    # Among high-precision thresholds, find one with best recall
                    best_precision_idx = precision_thresholds['recall'].idxmax()
                    best_precision_threshold = results_df.loc[best_precision_idx, 'threshold']
                else:
                    # If no threshold with precision >= 0.8, take the highest
                    best_precision_idx = results_df['precision'].idxmax()
                    best_precision_threshold = results_df.loc[best_precision_idx, 'threshold']
                
                # Find threshold with good recall (> 0.8 if available)
                recall_thresholds = results_df[results_df['recall'] >= 0.8]
                if len(recall_thresholds) > 0:
                    # Among high-recall thresholds, find one with best precision
                    best_recall_idx = recall_thresholds['precision'].idxmax()
                    best_recall_threshold = results_df.loc[best_recall_idx, 'threshold']
                else:
                    # If no threshold with recall >= 0.8, take the highest
                    best_recall_idx = results_df['recall'].idxmax()
                    best_recall_threshold = results_df.loc[best_recall_idx, 'threshold']
                
                recommendations[key] = {
                    'balanced': {
                        'threshold': best_f1_threshold,
                        'f1_score': results_df.loc[best_f1_idx, 'f1_score'],
                        'precision': results_df.loc[best_f1_idx, 'precision'],
                        'recall': results_df.loc[best_f1_idx, 'recall']
                    },
                    'precision_focused': {
                        'threshold': best_precision_threshold,
                        'f1_score': results_df.loc[best_precision_idx, 'f1_score'],
                        'precision': results_df.loc[best_precision_idx, 'precision'],
                        'recall': results_df.loc[best_precision_idx, 'recall']
                    },
                    'recall_focused': {
                        'threshold': best_recall_threshold,
                        'f1_score': results_df.loc[best_recall_idx, 'f1_score'],
                        'precision': results_df.loc[best_recall_idx, 'precision'],
                        'recall': results_df.loc[best_recall_idx, 'recall']
                    }
                }
            else:
                # For cases where we only have coverage percentage
                # Recommend thresholds for low, medium, and high coverage
                sorted_df = results_df.sort_values('coverage_pct')
                
                # Find thresholds for approximately 10%, 25%, and 50% coverage
                coverage_targets = [10, 25, 50]
                recommendations[key] = {}
                
                for target in coverage_targets:
                    closest_idx = (results_df['coverage_pct'] - target).abs().idxmin()
                    recommendations[key][f'{target}pct_coverage'] = {
                        'threshold': results_df.loc[closest_idx, 'threshold'],
                        'coverage_pct': results_df.loc[closest_idx, 'coverage_pct']
                    }
        
        # Save recommendations to file
        with open(f"{self.output_dir}/threshold_recommendations.txt", 'w') as f:
            f.write("RSSI Threshold Recommendations\n")
            f.write("=============================\n\n")
            
            for key, recs in recommendations.items():
                f.write(f"Analysis: {key}\n")
                f.write("-" * (10 + len(key)) + "\n")
                
                for scenario, values in recs.items():
                    f.write(f"  {scenario}:\n")
                    for metric, value in values.items():
                        f.write(f"    {metric}: {value}\n")
                f.write("\n")
        
        return recommendations

def run_threshold_analysis(merged_df, threshold_range=(-85, -65, 1), output_dir='output', plots_dir='plots'):
    """
    Run threshold sensitivity analysis on WiFi data
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged data with RSSI values
    threshold_range : tuple
        Range of thresholds to analyze (min, max, step)
    output_dir : str
        Directory to save output files
    plots_dir : str
        Directory to save plots
        
    Returns:
    --------
    analyzer : ThresholdSensitivityAnalyzer
        Analyzer object with results
    """
    print("Running threshold sensitivity analysis...")
    
    # Create analyzer
    analyzer = ThresholdSensitivityAnalyzer(
        output_dir=output_dir,
        plots_dir=plots_dir
    )
    
    # Run analysis for all data
    analyzer.analyze_thresholds(
        merged_df,
        rssi_column='rssi',
        threshold_range=threshold_range
    )
    
    # If transport modes are available, analyze by mode
    if 'predicted_mode' in merged_df.columns:
        analyzer.analyze_by_transport_mode(
            merged_df,
            rssi_column='rssi',
            threshold_range=threshold_range
        )
    
    # Get threshold recommendations
    recommendations = analyzer.recommend_thresholds()
    print("Threshold recommendations generated.")
    
    return analyzer, recommendations

if __name__ == "__main__":
    # Test the module with sample data
    try:
        print("Loading merged data...")
        try:
            merged_df = pd.read_csv('merged_wifi_location.csv')
            print(f"Loaded merged data with {len(merged_df)} rows")
            
            # Try to load transport modes if available
            try:
                transport_df = pd.read_csv('transport_modes.csv')
                print(f"Loaded transport modes with {len(transport_df)} rows")
                
                # Merge transport modes with merged data
                if 'time_window' in transport_df.columns and 'time_window' in merged_df.columns:
                    merged_df = pd.merge(
                        merged_df,
                        transport_df[['time_window', 'predicted_mode', 'transport_mode_code']],
                        on='time_window',
                        how='left'
                    )
                else:
                    print("Could not merge transport modes - missing time_window column")
            except FileNotFoundError:
                print("Transport modes not found. Running analysis without mode stratification.")
            
            # Run threshold analysis
            analyzer, recommendations = run_threshold_analysis(merged_df)
            
            print("\nRecommended thresholds:")
            for key, recs in recommendations.items():
                print(f"\nAnalysis: {key}")
                for scenario, values in recs.items():
                    print(f"  {scenario}:")
                    for metric, value in values.items():
                        print(f"    {metric}: {value}")
        
        except FileNotFoundError:
            print("Merged data not found. Please run data preprocessing first.")
        
    except Exception as e:
        print(f"Error during threshold analysis: {e}") 