import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

import argparse
from andon_stream import AndonStream
from local_stream_CPU import LocalStreamCPU
from local_stream_TPU import LocalStreamTPU

# Cloud is implemented using google collab notebook, where the class is defined in the same notebook

class DataSampler:
    """Samples data from a detection stream at regular intervals."""
    def __init__(self, stream, sample_interval=0.1):
        self.stream = stream
        self.sample_interval = sample_interval
        self.samples = []
        
    def sample(self, num_samples=100, require_detection=True):
        """
        Collect specified number of samples from the stream.
        
        Args:
            num_samples: Number of valid samples to collect
            require_detection: If True, only collect samples with successful detections
            
        Returns:
            list: Collected samples with performance metrics
        """
        print(f"Starting sampling process for {num_samples} samples...")
        collected = 0
        
        while collected < num_samples:

            detected = self.stream.get_latest_detection_status()


            if detected is False:
                time.sleep(self.sample_interval)
                continue

            # Get latest inference time
            inference_time = self.stream.get_latest_inference_time()
            
            # Get latest end-to-end time
            end_to_end_time = self.stream.get_latest_end_to_end_time()

            
            # Skip if no valid data
            if inference_time is None or end_to_end_time is None:
                time.sleep(self.sample_interval)
                continue
            
            # Create sample record
            sample = {
                'timestamp': datetime.now().isoformat(),
                'inference_time': inference_time,
                'end_to_end_time': end_to_end_time
            }
            
            self.samples.append(sample)
            collected += 1
            
            if collected % 10 == 0:
                print(f"Collected {collected}/{num_samples} samples")
                
            time.sleep(self.sample_interval)
            
        print(f"Sampling complete. Collected {len(self.samples)} samples.")
        return self.samples

class DataAnalyzer:
    """
    Analyzes performance data from detection samples and
    generates visualizations and reports.
    """
    def __init__(self, samples, deployment_type="andon"):
        self.samples = samples
        self.deployment_type = deployment_type
        self.df = pd.DataFrame(samples)
        self.results = self._analyze()
        
    def _analyze(self):
        """Analyze the collected samples and compute statistics."""
        if len(self.df) == 0:
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
        
    def output_figures(self, output_dir="./results"):
        """Generate visualization figures for the data."""
        import os
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
        
        # Plot end-to-end time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['end_to_end_time'], bins=20, alpha=0.7, color='green')
        plt.title(f'End-to-End Latency Distribution - {self.deployment_type}')
        plt.xlabel('End-to-End Time (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/{self.deployment_type}_end_to_end_time_{timestamp}.png")
        
        # Print summary to console
        self._print_summary()
        
    def output_csv(self, output_dir="./results"):
        """Save the raw data and analysis results to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        self.df.to_csv(f"{output_dir}/{self.deployment_type}_raw_data_{timestamp}.csv", index=False)
        
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
        print(f"Results saved to {output_dir}")
        
    def _print_summary(self):
        """Print a summary of the analysis results to the console."""
        print("\n" + "="*50)
        print(f"PERFORMANCE SUMMARY - {self.deployment_type.upper()}")
        print("="*50)
        print(f"Samples collected: {self.results['sample_count']}")
        
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
    parser = argparse.ArgumentParser(description='Latency Test for Object Detection Systems')
    parser.add_argument('--test', type=str, required=True, choices=['andon', 'cpu', 'tpu'],
                        help='Test type to run (andon, cpu, tpu, cloud)')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to collect (default: 100)')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Sampling interval in seconds (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='/app/results',
                        help='Directory to save output files (default: /app/results)')
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()

        if args.output_dir == "/app/results":
            os.makedirs(args.output_dir, exist_ok=True)

        else: 
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize appropriate stream based on test type
        stream = None
        if args.test == "andon":
            stream = AndonStream()
            print("Starting Andon System stream...")
            stream.run()
        elif args.test == "cpu":
            stream = LocalStreamCPU()
            print("Starting local CPU stream...")
            stream.run()

        elif args.test == "tpu":
            stream = LocalStreamTPU()
            print("Starting local TPU stream...")
            stream.run()

        elif args.test == "cloud":
            # Placeholder for Cloud stream implementation
            print("Cloud implementation not available yet")
            exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        data_sampler = DataSampler(stream, sample_interval=args.interval)
        data = data_sampler.sample(num_samples=args.samples)
        
        analyzer = DataAnalyzer(data, deployment_type=args.test)
        analyzer.output_figures(output_dir=args.output_dir)
        analyzer.output_csv(output_dir=args.output_dir)
        
    finally:
        # Clean up
        if stream:
            print("Stopping stream...")
            stream.stop()
            