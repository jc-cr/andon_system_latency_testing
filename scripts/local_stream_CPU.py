#!/usr/bin/env python3
"""
LocalStreamCPU Class for Performance Testing

This is an integration class that implements the same interface as AndonStream
and LocalStreamTPU for the performance testing framework, but uses CPU for inference.
"""

import logging
import numpy as np
import time
import cv2
import os
from threading import Thread, Lock, Event
import tensorflow as tf

class LocalStreamCPU:
    """
    Local CPU stream implementation for performance testing.
    
    This class follows the same interface as AndonStream and LocalStreamTPU
    to work with the existing performance testing framework.
    """
    def __init__(self, model_path=None, label_path=None, camera_id=0):
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.info("Initializing LocalStreamCPU")
        
        # Print TensorFlow version for debugging
        self.logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Default paths for TFLite model (not EdgeTPU version)
        if model_path is None:
            model_path = '/app/models/tf2_ssd_mobilenet_v2_coco17_ptq.tflite'
        if label_path is None:
            label_path = '/app/models/coco_labels.txt'
            
        # Store configuration
        self.model_path = model_path
        self.label_path = label_path
        self.camera_id = camera_id
        self.width = 640
        self.height = 480
        self.threshold = 0.6
        self.person_class_id = 0  # In COCO dataset, person class ID is 0
        
        # Performance metrics
        self.last_inference_time = 0
        self.last_capture_time = 0
        self.last_processing_time = 0
        
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
        self.input_details = None
        self.output_details = None
    
    def run(self):
        """Start the streaming and detection process."""
        try:
            self.logger.info("Starting LocalStreamCPU")
            
            # Open camera
            self.logger.info(f"Opening camera {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
                
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Log camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
                
            # Load the TFLite model
            self.logger.info(f"Loading model: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Debug output tensor shapes to help understand the model inputs/outputs
            self.logger.debug(f"Input details: {self.input_details}")
            self.logger.debug(f"Output details: {self.output_details}")
            
            # Load labels if available
            if os.path.exists(self.label_path):
                self.labels = self._load_labels(self.label_path)
                self.logger.debug(f"Loaded {len(self.labels)} labels")
            else:
                self.logger.warning(f"Label file not found: {self.label_path}")
                self.labels = {0: 'person'}  # Fallback
            
            # Start processing thread
            self.running.set()
            self.thread = Thread(target=self._processing_thread)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("LocalStreamCPU started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start LocalStreamCPU: {e}", exc_info=True)
            if self.cap and self.cap.isOpened():
                self.cap.release()
            return False
    
    def _load_labels(self, path):
        """Load label file and return a dictionary mapping indices to labels."""
        with open(path, 'r') as f:
            lines = f.readlines()
        labels = {}
        for i, line in enumerate(lines):
            labels[i] = line.strip()
        return labels
    
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
                
                # Log all detections for debugging
                if detections:
                    detection_info = []
                    for d in detections:
                        class_id, score = d[1], d[2]
                        class_name = self.labels.get(class_id, f"Unknown({class_id})")
                        detection_info.append(f"{class_name}: {score:.2f}")
                    self.logger.debug(f"Detections: {detection_info}")
                
                # Check for person detections
                detected = False
                for detection in detections:
                    class_id = int(detection[1])
                    confidence = detection[2]
                    if class_id == self.person_class_id and confidence > self.threshold:
                        detected = True
                        self.logger.debug(f"Person detected with confidence: {confidence:.2f}")
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
        Process a frame using TFLite model on CPU.
        
        Args:
            frame: The image frame to process
            
        Returns:
            (detections, inference_time): List of detections and inference time in ms
        """
        try:
            # Get input size from model
            input_shape = self.input_details[0]['shape']
            input_height, input_width = input_shape[1], input_shape[2]
            
            # Prepare the frame for the model
            input_frame = cv2.resize(frame, (input_width, input_height))
            # Convert to RGB as models are typically trained on RGB
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            
            # Use UINT8 as required by the model
            input_frame = input_frame.astype(np.uint8)
            
            # Add batch dimension
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_frame)
            
            # Measure inference time
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            
            # Get output tensors
            # Extract the outputs based on their indices
            detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            
            # Log shapes for debugging (only at DEBUG level to avoid too much output)
            self.logger.debug(f"detection_boxes shape: {detection_boxes.shape}")
            self.logger.debug(f"detection_classes shape: {detection_classes.shape}")
            self.logger.debug(f"detection_scores shape: {detection_scores.shape}")
            
            # Process results
            detections = []
            
            # Based on the logs, we have:
            # - detection_boxes: (1, 20) - Single float values, not coordinate arrays
            # - detection_classes: (1, 20, 4) - Each detection has 4 class probabilities
            # - detection_scores: (1,) - Single overall confidence
            
            # Check if the overall detection score passes our threshold
            if detection_scores.shape == (1,) and detection_scores[0] > self.threshold:
                overall_score = float(detection_scores[0])
                self.logger.debug(f"Overall detection score: {overall_score}")
                
                # Look through the class probabilities for person detections
                if len(detection_classes.shape) == 3 and detection_classes.shape[2] >= 1:
                    num_detections = detection_classes.shape[1]
                    
                    for i in range(num_detections):
                        # Get the probability for person class (index 0)
                        # Access with safety checks
                        if detection_classes.shape[2] > self.person_class_id:
                            person_prob = float(detection_classes[0, i, self.person_class_id])
                            
                            # If person probability is high enough
                            if person_prob > self.threshold:
                                self.logger.debug(f"Detected person with probability: {person_prob:.4f}")
                                
                                # We don't have proper bounding box coordinates, so use a simplified approach
                                # Create a detection with a centered box covering 80% of the frame
                                
                                # Calculate a reasonable bounding box (centered, 80% of frame)
                                width = frame.shape[1]
                                height = frame.shape[0]
                                box_width = int(width * 0.8)
                                box_height = int(height * 0.8)
                                xmin = int((width - box_width) / 2)
                                ymin = int((height - box_height) / 2)
                                xmax = xmin + box_width
                                ymax = ymin + box_height
                                
                                # Add detection
                                detections.append((
                                    i,                     # detection ID
                                    self.person_class_id,  # class ID (person)
                                    person_prob,           # confidence score
                                    (xmin, ymin, xmax, ymax)  # estimated bounding box
                                ))
            
            return detections, inference_time
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
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
        Get the end-to-end latency including capture, processin
        
        Returns:
            float: End-to-end latency in milliseconds or None if not available
        """
        with self.lock:
            if self.last_inference_time <= 0:
                return None
            
            # Calculate total latency (capture + processing)
            total_time = self.last_capture_time + self.last_processing_time
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
        self.logger.info("Stopping LocalStreamCPU")
        self.running.clear()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        self.logger.info("LocalStreamCPU stopped")

if __name__ == "__main__":
    # Simple test code
    stream = LocalStreamCPU()
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