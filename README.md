# TODO
Worker: handle change request
Producer: Implement change-log should worker config change  --> implement changelog dict (with stack of changelogs for each worker)
Implement program that puts frames back together
Implement QoS Solutions

Add Super resolution mode (up-scaling)
Find more efficient way to load YOLO model, i.e. load with GPU maybe

## How to send numpy arrays using 0MQ efficiently:
https://pyzmq.readthedocs.io/en/latest/howto/serialization.html#example-numpy-arrays

# Dependencies
## How to build `requirements.txt`
```pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install msgpack

pip freeze > requirements.txt
```

Add on top of requirements.txt
```
-f https://download.pytorch.org/whl/cu124/torch
-f https://download.pytorch.org/whl/cu124/torchvision
-f https://download.pytorch.org/whl/cu124/torchaudio
```

# Description
This is a distributed Parallel Pipeline using active inference to dynamically change the size/frequency/complexity of the tasks 
This is an edge node for a distributed systems. The distributed system consists of multiple edge nodes and a single controller.

## Worker
The edge nodes constantly receive a video stream either via a network or simulated from a source video file. The edge nodes 
perform computations upon the individual frames of the stream and send the result to a different node (such as the controller) where the 
resulting computed frames are stitched together resulting in the new output video stream.

Computations upon the original video stream include:
- YOLOv11 General Object Detection with regular bounding boxes
- YOLOv11 Arial View Object Detection with oriented bounding boxes
- Resolution up-scaling

### Task Splitting
In order to improve overall performance and to harness maximum computational resources from the distributed system, the 
workload/task is split between all the available edge nodes. 

### Measures to uphold Quality of Service (QoS)
Should the computational resources of the system are not enough to uphold certain Service level objectives (SLOs), 
i.e. processing time for x amount frames, energy consumption, etc then the edge node has some options to reduce the 
required computational load. These include:

- Reducing Quality: Switch to a lower grade YOLOv11 model, Reduce up-scaling quality.
- Reduce Source video FPS
- Reduce Source video Quality

### Active Inference
In order to choose which measure is used to uphold QoS/SLOs, the edge node uses Active Inference.

## Producer
The producer creates the tasks and sends them to the workers upon receiving a request.

# Implementation
## Active Inference Model
Use [pymdp](https://github.com/infer-actively/pymdp)
## Network Communication
### Control Channel
- ZeroMQ: REQ-REP pattern for requesting changes at the controller. PUB-SUB pattern for Controller to broadcast changes (TODO: either TCP or PGM/EPGM)
### Data Channel
- Local: ``multiprocessing.connection: Client & Server``
- Network Streaming: ``zeroMQ`` Radio-Dish pattern (UDP)
### Serialization
**msgpack vs pickle:**
- performance: Similar performance for larger data (i.e. video frames). Pickle faster on smaller data.
- size: msgpack is always slightly smaller

# SLO Ideas
Select at least 2 Types of SLO
1. **Performance:** SLOs that causes the underlying stream to be offloaded, quality reduced, etc
2. **Quality Goal:** SLOs that make sure a certain standard of quality is upheld. The Workers/Controller should notice if the quality is compromised and increase it accordingly.
## General Ideas
- Latency
- Accuracy
- Resource Efficiency

## Specific SLOs:
### Size of Frame Buffer
- Frame Buffer must not exceed a certain size
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
