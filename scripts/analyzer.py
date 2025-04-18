"""
Analysis code for processing raw timing data from different model deployments.
Reads existing CSV files and generates performance visualizations and reports.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl


# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 8)  # Larger default figure size
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300

# Set font with fallbacks to system fonts
plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 18  # Title font size
plt.rcParams['axes.labelsize'] = 16  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 14  # Legend font size
plt.rcParams['figure.titlesize'] = 20  # Figure title size


class DataAnalyzer:
    """
    Analyzes performance data from detection samples and
    generates visualizations and reports.
    """
    def __init__(self, csv_path, deployment_type=None):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            csv_path (str): Path to the raw data CSV file
            deployment_type (str, optional): Type of deployment. If None, it will be
                                             extracted from the CSV filename.
        """
        self.csv_path = csv_path
        
        # Extract deployment type from filename if not provided
        if deployment_type is None:
            deployment_type = self._extract_deployment_type(csv_path)
        
        self.deployment_type = deployment_type
        
        # Read the data from CSV
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Successfully loaded data from {csv_path}")
            print(f"Found {len(self.df)} samples")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)
            
        self.results = self._analyze()
        
    def _extract_deployment_type(self, csv_path):
        """Extract deployment type from the CSV filename."""
        filename = os.path.basename(csv_path)
        
        # Common deployment types
        deployment_types = ["andon", "cpu", "tpu", "cloud", "cloud_gpu"]
        
        for dt in deployment_types:
            if dt in filename.lower():
                return dt
                
        # If no match found, use a generic name
        return "deployment"
        
    def _analyze(self):
        """Analyze the loaded data and compute statistics."""
        if len(self.df) == 0:
            print("Warning: No data found in the CSV file.")
            return None
            
        # Calculate basic statistics for inference time
        inference_stats = {
            'mean': self.df['inference_time'].mean(),
            'median': self.df['inference_time'].median(),
            'std': self.df['inference_time'].std(),
            'min': self.df['inference_time'].min(),
            'max': self.df['inference_time'].max(),
            'percentile_95': np.percentile(self.df['inference_time'], 95)
        }
        
        # Calculate basic statistics for end-to-end time
        end_to_end_stats = {
            'mean': self.df['end_to_end_time'].mean(),
            'median': self.df['end_to_end_time'].median(),
            'std': self.df['end_to_end_time'].std(),
            'min': self.df['end_to_end_time'].min(),
            'max': self.df['end_to_end_time'].max(),
            'percentile_95': np.percentile(self.df['end_to_end_time'], 95)
        }
        
        return {
            'deployment_type': self.deployment_type,
            'sample_count': len(self.df),
            'inference_stats': inference_stats,
            'end_to_end_stats': end_to_end_stats
        }
        
    def output_figures(self, output_dir=None):
        """Generate visualization figures for the data."""
        if output_dir is None:
            # Use the directory of the input file as the default output directory
            output_dir = os.path.dirname(self.csv_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot inference time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['inference_time'], bins=20, alpha=0.7, color='blue')
        plt.title(f'Inference Time Distribution - {self.deployment_type}')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/{self.deployment_type}_inference_time_{timestamp}.png")
        print(f"Saved inference time plot to {output_dir}/{self.deployment_type}_inference_time_{timestamp}.png")
        
        # Plot end-to-end time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['end_to_end_time'], bins=20, alpha=0.7, color='green')
        plt.title(f'End-to-End Latency Distribution - {self.deployment_type}')
        plt.xlabel('End-to-End Time (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/{self.deployment_type}_end_to_end_time_{timestamp}.png")
        print(f"Saved end-to-end time plot to {output_dir}/{self.deployment_type}_end_to_end_time_{timestamp}.png")
        
        # Print summary to console
        self._print_summary()
        
    def output_csv(self, output_dir=None):
        """Save the analysis results to a CSV file."""
        if output_dir is None:
            # Use the directory of the input file as the default output directory
            output_dir = os.path.dirname(self.csv_path)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis results
        results_df = pd.DataFrame({
            'metric': [
                'sample_count',
                'inference_mean', 'inference_median', 'inference_std', 'inference_min', 'inference_max', 'inference_95p',
                'e2e_mean', 'e2e_median', 'e2e_std', 'e2e_min', 'e2e_max', 'e2e_95p'
            ],
            'value': [
                self.results['sample_count'],
                self.results['inference_stats']['mean'],
                self.results['inference_stats']['median'],
                self.results['inference_stats']['std'],
                self.results['inference_stats']['min'],
                self.results['inference_stats']['max'],
                self.results['inference_stats']['percentile_95'],
                self.results['end_to_end_stats']['mean'],
                self.results['end_to_end_stats']['median'],
                self.results['end_to_end_stats']['std'],
                self.results['end_to_end_stats']['min'],
                self.results['end_to_end_stats']['max'],
                self.results['end_to_end_stats']['percentile_95']
            ]
        })
        
        results_df.to_csv(f"{output_dir}/{self.deployment_type}_analysis_{timestamp}.csv", index=False)
        print(f"Saved analysis results to {output_dir}/{self.deployment_type}_analysis_{timestamp}.csv")
        
    def _print_summary(self):
        """Print a summary of the analysis results to the console."""
        print("\n" + "="*50)
        print(f"PERFORMANCE SUMMARY - {self.deployment_type.upper()}")
        print("="*50)
        print(f"Samples analyzed: {self.results['sample_count']}")
        
        print("\nINFERENCE TIME STATISTICS (ms):")
        print(f"  Mean:      {self.results['inference_stats']['mean']:.2f}")
        print(f"  Median:    {self.results['inference_stats']['median']:.2f}")
        print(f"  Std Dev:   {self.results['inference_stats']['std']:.2f}")
        print(f"  Min:       {self.results['inference_stats']['min']:.2f}")
        print(f"  Max:       {self.results['inference_stats']['max']:.2f}")
        print(f"  95th Perc: {self.results['inference_stats']['percentile_95']:.2f}")
        
        print("\nEND-TO-END LATENCY STATISTICS (ms):")
        print(f"  Mean:      {self.results['end_to_end_stats']['mean']:.2f}")
        print(f"  Median:    {self.results['end_to_end_stats']['median']:.2f}")
        print(f"  Std Dev:   {self.results['end_to_end_stats']['std']:.2f}")
        print(f"  Min:       {self.results['end_to_end_stats']['min']:.2f}")
        print(f"  Max:       {self.results['end_to_end_stats']['max']:.2f}")
        print(f"  95th Perc: {self.results['end_to_end_stats']['percentile_95']:.2f}")
        print("="*50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze latency data from CSV files')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the raw data CSV file')
    parser.add_argument('--deployment', type=str, default=None,
                        help='Deployment type label (default: auto-detected from filename)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files (default: same as input file)')
    return parser.parse_args()


def main():
    """Main function to analyze latency data from CSV files."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    # Create analyzer and process the data
    analyzer = DataAnalyzer(args.input, deployment_type=args.deployment)
    analyzer.output_figures(output_dir=args.output_dir)
    analyzer.output_csv(output_dir=args.output_dir)


if __name__ == "__main__":
    main()