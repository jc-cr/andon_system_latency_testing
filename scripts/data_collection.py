"""
Logging code for streaming scripts. Logs key times from different model deployments.
"""


import argparse
import time
import numpy as np
import pandas as pd
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
        
        
    finally:
        # Clean up
        if stream:
            print("Stopping stream...")
            stream.stop()
            