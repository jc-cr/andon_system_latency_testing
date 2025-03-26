#!/usr/bin/env python3
"""
LocalStreamTPU Class for Performance Testing

This is an integration class that implements the same interface as AndonStream
for the performance testing framework.
"""

import logging
import numpy as np
import time
import cv2
import os
from threading import Thread, Lock, Event

# PyCoral imports
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class LocalStreamTPU:
    """
    Local Edge TPU stream implementation for performance testing.
    
    This class follows the same interface as AndonStream to work with
    the existing performance testing framework.
    """
    def __init__(self, model_path=None, label_path=None, camera_id=0):
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.info("Initializing LocalStreamTPU")
        
        # Default paths
        if model_path is None:
            model_path = '/app/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'
        if label_path is None:
            label_path = '/app/models/coco_labels.txt'
            
        # Store configuration
        self.model_path = model_path
        self.label_path = label_path
        self.camera_id = camera_id
        self.width = 640
        self.height = 480
        self.threshold = 0.5
        self.person_class_id = 0  # In COCO dataset, person class ID is 0
        
        # Performance metrics
        self.last_inference_time = 0
        self.last_capture_time = 0
        self.last_processing_time = 0
        self.last_transmission_time = 3.0  # Simulated network delay (ms)
        
        # Detection results
        self.detected = False
        self.frame = None
        self.detections = []
        
        # Thread management
        self.running = Event()
        self.lock = Lock()
        self.thread = None
        
        # Initialize camera and model later in run()
        self.cap = None
        self.interpreter = None
        self.labels = None
    
    def run(self):
        """Start the streaming and detection process."""
        try:
            self.logger.info("Starting LocalStreamTPU")
            
            # Open camera
            self.logger.info(f"Opening camera {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Initialize Edge TPU model
            self.logger.info(f"Loading model: {self.model_path}")
            self.interpreter = make_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            
            # Load labels if available
            if os.path.exists(self.label_path):
                self.labels = read_label_file(self.label_path)
                self.logger.info(f"Loaded {len(self.labels)} labels")
            else:
                self.logger.warning(f"Label file not found: {self.label_path}")
                self.labels = {0: 'person'}  # Fallback
            
            # Start processing thread
            self.running.set()
            self.thread = Thread(target=self._processing_thread)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("LocalStreamTPU started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start LocalStreamTPU: {e}", exc_info=True)
            if self.cap and self.cap.isOpened():
                self.cap.release()
            return False
    
    def _processing_thread(self):
        """Background thread for frame processing."""
        while self.running.is_set():
            try:
                # Measure frame capture time
                capture_start = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                # Calculate capture time in milliseconds
                capture_time = (time.time() - capture_start) * 1000
                
                # Process the frame
                process_start = time.time()
                detections, inference_time = self._process_frame(frame)
                process_time = (time.time() - process_start) * 1000
                
                # Check for person detections
                detected = False
                for d in detections:
                    if d.id == self.person_class_id and d.score > self.threshold:
                        detected = True
                        break
                
                # Update state with thread safety
                with self.lock:
                    self.frame = frame
                    self.detections = detections
                    self.detected = detected
                    self.last_inference_time = inference_time
                    self.last_capture_time = capture_time
                    self.last_processing_time = process_time
                
                # Short delay to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        """
        Process a frame using the Edge TPU.
        
        Args:
            frame: The image frame to process
            
        Returns:
            (detections, inference_time): List of detections and inference time in ms
        """
        try:
            # MobileNet SSD v2 expects 300x300 input
            input_size = common.input_size(self.interpreter)
            
            # Prepare the frame for the model
            _, scale = common.set_resized_input(
                self.interpreter,
                input_size,
                lambda size: cv2.resize(frame, size)
            )
            
            # Measure inference time
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            
            # Get detections
            detections = detect.get_objects(self.interpreter, self.threshold, scale)
            
            return detections, inference_time
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            return [], 0

    def get_latest_inference_time(self):
        """
        Get the latest inference time (time taken for model to process image).
        
        Returns:
            float: Inference time in milliseconds or None if not available
        """
        with self.lock:
            return self.last_inference_time if self.last_inference_time > 0 else None

    def get_latest_end_to_end_time(self):
        """
        Get the end-to-end latency including capture, processing, and transmission time.
        
        Returns:
            float: End-to-end latency in milliseconds or None if not available
        """
        with self.lock:
            if self.last_inference_time <= 0:
                return None
            
            # Calculate total latency (capture + processing + transmission)
            total_time = self.last_capture_time + self.last_processing_time + self.last_transmission_time
            return total_time

    def get_latest_detection_status(self):
        """
        Get the latest detection status.
        
        Returns:
            bool: True if person detected, False otherwise
        """
        with self.lock:
            return self.detected
    
    def stop(self):
        """Stop streaming and release resources."""
        self.logger.info("Stopping LocalStreamTPU")
        self.running.clear()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        self.logger.info("LocalStreamTPU stopped")

if __name__ == "__main__":
    # Simple test code
    stream = LocalStreamTPU(camera_id=0)
    if stream.run():
        print("Stream started successfully")
        
        try:
            # Display basic stats for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                time.sleep(1)
                print(f"Detection: {stream.get_latest_detection_status()}")
                print(f"Inference time: {stream.get_latest_inference_time()} ms")
                print(f"End-to-end time: {stream.get_latest_end_to_end_time()} ms")
                print("-" * 30)
                
        finally:
            stream.stop()
    else:
        print("Failed to start stream")
