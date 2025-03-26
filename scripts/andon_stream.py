#!/usr/bin/env python3
"""
Andon System Stream Implementation

Adapts the existing Coral Micro with TOF sensor code to match
the format expected by the dynamic test visualizer.
"""

import logging
import numpy as np
import time
import threading
import json
import base64
import requests
from threading import Thread, Lock, Event

class AndonStream:
    """
    Andon system (Coral Micro) stream implementation.
    
    Connects to the Coral Micro board via HTTP and retrieves
    camera images, detection data, and depth information.
    """
    def __init__(self, ip="10.10.10.1", poll_interval=0.033):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Andon system stream")
        
        # Configuration
        self.ip = ip
        self.poll_interval = poll_interval
        
        # Data storage
        self.frame = None
        self.detected = False
        
        # Stats
        self.last_timestamp = 0              # When the log was generated on device
        self.last_inference_time = 0         # Model inference time on device
        self.last_depth_estimation_time = 0  # Depth estimation time on device
        self.image_capture_timestamp = 0     # When the image was captured on device
        self.last_receive_time = 0           # When we processed the data
        self.last_transmission_time = 0      # Round-trip network transmission time
        self.depth = 0.0
        
        # Thread safety
        self.lock = Lock()
        self.running = Event()
        self.thread = None
    
    def run(self):
        """Start the Andon system stream """
        try:
            self.running.set()
            
            # Start polling thread
            self.thread = Thread(target=self._update_thread)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Andon system stream started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Andon system stream: {e}", exc_info=True)
            return False
    
    def _update_thread(self):
        """Background thread for updating detection and depth data."""
        while self.running.is_set():
            try:
                # Fetch data from device
                data = self._fetch_data()
                if data:
                    self._process_data(data)
                
                # Sleep to maintain frame rate
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Andon update thread: {e}")
                time.sleep(0.1)  # Short delay on error
    
    def _fetch_data(self):
        """Fetch data from the device using JSON-RPC."""
        try:
            # Record when we're making the request
            request_time = int(time.time() * 1000)
            
            payload = {
                'id': 1,
                'jsonrpc': '2.0',
                'method': 'tx_logs_to_host',
                'params': {}
            }
            
            # Use a shorter timeout to prevent blocking
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=1.0
            )
            
            # Record when we received the response
            receive_time = int(time.time() * 1000)
            
            # Calculate transmission time
            transmission_time = receive_time - request_time
            
            # Update transmission time with thread safety
            with self.lock:
                self.last_transmission_time = transmission_time
            
            if response.status_code != 200:
                self.logger.warning(f"Data fetch HTTP error: {response.status_code}")
                return None
            
            result = response.json()
            if 'error' in result:
                self.logger.warning(f"RPC error: {result['error']}")
                return None
                
            if 'result' not in result:
                self.logger.warning("No result in response")
                return None
                
            return result['result']
            
        except requests.exceptions.Timeout:
            self.logger.warning("Request to device timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def _process_data(self, data):
        """Process received data from the device."""
        try:
            # Record when this data was received
            receive_time = int(time.time() * 1000)
            
            # Parse basic metadata
            timestamp = data.get('log_timestamp_ms', 0)
            detection_count = data.get('detection_count', 0)
            inference_time = data.get('inference_time_ms', 0)
            depth_estimation_time = data.get('depth_estimation_time_ms', 0)
            image_capture_timestamp = data.get('image_capture_timestamp_ms', 0)
            
            
            if 'depths' in data:
                depth_bytes = base64.b64decode(data['depths'])
                depth_data = self._parse_depths(depth_bytes, detection_count)

                if depth_data and depth_data['count'] > 0:
                    self.depth = depth_data['depths'][0]

            # Prepare variables for detection data
            frame = None
            detected = False
            
            if detection_count > 0 and 'detections' in data:
                detected = True
            
            # Update state with thread safety
            with self.lock:
                self.last_timestamp = timestamp
                self.last_inference_time = inference_time
                self.last_depth_estimation_time = depth_estimation_time
                self.image_capture_timestamp = image_capture_timestamp
                self.last_receive_time = receive_time
                
                if frame is not None:
                    self.frame = frame
                
                self.detected = detected
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")

    def _parse_depths(self, depth_bytes, count):
        """Parse binary depth data."""
        try:
            # Parse as array of floats
            depths = np.frombuffer(depth_bytes, dtype=np.float32, count=count)
            return {
                'count': count,
                'depths': depths.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing depth data: {e}")
            return {'count': 0, 'depths': []}

    def get_latest_inference_time(self):
        """
        Get the latest inference time (time taken for model to process image).
        
        Returns:
            float: Inference time in milliseconds or None if not available
        """
        with self.lock:
            if self.last_inference_time <= 0:
                return None
            return self.last_inference_time

    def get_latest_end_to_end_time(self):
        """
        Get the end-to-end latency including transmission time but excluding depth estimation.
        
        This calculates:
        1. Time from image capture to result processing on device (minus depth estimation)
        2. Time for network transmission from device to client
        
        Returns:
            float: End-to-end latency in milliseconds or None if not available
        """
        with self.lock:
            if self.last_inference_time <= 0:
                return None
            
            # Start with base inference time
            total_time = self.last_inference_time
            
            # Add transmission time if available
            if self.last_transmission_time > 0:
                total_time += self.last_transmission_time
                
            # Try to calculate more accurate device processing time
            if self.last_timestamp > 0 and self.image_capture_timestamp > 0:
                # On-device processing time excluding depth estimation
                device_processing_time = self.last_timestamp - self.image_capture_timestamp
                
                if self.last_depth_estimation_time > 0:
                    # Don't let it go negative
                    depth_time = min(self.last_depth_estimation_time, device_processing_time)
                    device_processing_time -= depth_time
                
                # Use the calculated processing time instead of just inference time
                # Add the transmission time to get the full end-to-end latency
                return device_processing_time + self.last_transmission_time
            
            # Fallback to basic calculation
            return total_time

    def get_latest_detection_status(self):
        """
        Get the latest detection status.
        
        Returns:
            bool: True if object detected, False otherwise
        """
        with self.lock:
            return self.detected
    
    def get_stats(self):
        """
        Get all timing statistics as a dictionary.
        This is safer than accessing internal variables directly.
        """
        with self.lock:
            return {
                "inference_time": self.last_inference_time,
                "depth_estimation_time": self.last_depth_estimation_time,
                "device_processing_time": self.last_timestamp - self.image_capture_timestamp if self.last_timestamp > 0 and self.image_capture_timestamp > 0 else 0,
                "transmission_time": self.last_transmission_time,
                "detected": self.detected,
                "depth": self.depth
            }
    
    def stop(self):
        """Stop streaming and release resources."""
        self.running.clear()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        self.logger.info("Andon system stream stopped")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Stream time data if detection
    stream = AndonStream()
    stream.run()

    print("Streaming Andon system data")

    # Log data until user interrupts
    try:
        while True:
            time.sleep(1)
            # Get stats in a thread-safe way without holding lock during printout
            stats = stream.get_stats()
            
            if stats["detected"]:
                # Calculate components
                device_processing = stats["device_processing_time"]
                depth_time = stats["depth_estimation_time"]
                processing_minus_depth = max(0, device_processing - depth_time)
                
                # Get end-to-end time using the method to ensure consistency
                end_to_end = stream.get_latest_end_to_end_time()
                
                print("\n--- Detection Data ---")
                print(f"Detection: {stats['detected']}")
                print(f"Depth: {stats['depth']} m")
                print(f"Model Inference Time: {stats['inference_time']} ms")
                print(f"Device Processing Time: {device_processing} ms")
                print(f"Depth Estimation Time: {depth_time} ms")
                print(f"Processing (minus depth): {processing_minus_depth} ms")
                print(f"Network Transmission Time: {stats['transmission_time']} ms")
                print(f"Total End-to-End Time: {end_to_end} ms")
                print("---------------------")
            
    except KeyboardInterrupt:
        pass

    finally:
        print("Stopping Andon system stream")
        stream.stop()