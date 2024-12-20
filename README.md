# TODO
Find more efficient way to load YOLO model, i.e. load with GPU maybe
Extract OBB Boxes from yolo-inference result and draw them
Eventuell modus der Frames skipped um Real-Time aufrecht zu erhalten


# SLO Ideas
## Generel Ideas
- Latency
- Accuracy
- Resource Efficiency

# Specific SLOs:


### Accuracy-Driven SLOs:
- Bounding Box Confidence Threshold: Ensure that detected objects have confidence scores above a certain threshold. For example, bounding boxes should only be displayed if YOLO predicts an object with ≥90% confidence.
- False Positive Rate (FPR): Limit the number of incorrect bounding boxes (false positives) to maintain the reliability of displayed results.
- Missed Detections: Ensure that the rate of undetected objects in the frame is below a certain percentage, especially for critical objects.
### Performance-Driven SLOs:
- Frame Processing Time: Cap the processing time per frame to maintain real-time constraints, e.g., ≤50ms/frame.
- Device Utilization Balance: Ensure that computational workloads are evenly distributed across devices, avoiding overloading any single device.
- Dropped Frames: Minimize the number of dropped frames, ensuring a smooth video stream.
### Quality-Driven SLOs:
- Bounding Box Smoothness: Ensure bounding box coordinates for tracked objects do not jitter between frames unless there is a significant change in the object's position.
- Detection Latency Consistency: Maintain a consistent latency between object appearance in the frame and the inference result being displayed.
### Robustness SLOs:
- Recovery Time from Failure: Define a maximum recovery time (e.g., ≤2 seconds) if a device goes offline or the system faces temporary overload.
- Dynamic Load Adjustment: Devices should adjust workloads dynamically within a predefined response time (e.g., 500ms) when performance drops below an acceptable level.
### Monitoring and Feedback:
- Error Rates: Track and limit YOLO's internal errors, such as tensor misalignment or model loading errors.
- Output Stability Across Devices: Ensure consistent results across distributed devices processing overlapping frames or regions.
### Object Counting SLOs
- Counting Accuracy: Ensure the object count for each frame has a minimal margin of error (e.g., ≤5% deviation from ground truth for known test cases).
- Counting Consistency: Maintain consistency in object counts across consecutive frames, especially for stationary objects, to avoid sudden spikes or drops.
- Category-Specific Thresholds: Set thresholds for accuracy or count deviations for critical object categories, such as vehicles in traffic or people in crowds.

# Implementation
## 1. Implement Monitoring and Feedback
- Device metrics: CPU/GPU utilization, memory usage, battery level.
- Network metrics: Bandwidth, latency.
- Inference metrics: Accuracy, confidence scores.

Using Python, you can monitor system metrics with libraries like `psutil` or `pySMART`.
## 2. 