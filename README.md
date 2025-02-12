# TODO
Properly close all programs after task generation is done.
Test the collector


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
This is a distributed Parallel Pipeline using active inference to dynamically change the size/frequency/complexity of the tasks.
The goal is to run computations on a live video stream in parallel.
It consists of 3 program types:

- Producer
- Worker
- Collector

This is an edge node for a distributed systems. The distributed system consists of multiple edge nodes and a single controller.
## Producer
The producer constantly creates work (frames of the source video stream) in the form of multiple `Task` objects. A `Task` is `numpy.ndarray` with a `id` and `type`. 
For example, when running inference on a video stream the producer would produce tasks such as:
```
Task:
    id = 10
    type = 'INFERNCE'
    data = np.ndarray
```
The produces tasks are then buffered and ready to be requested via Work-API

### Elasticity
The Producer tries to ensure Quality of Service (QoS) by providing certain elasticity features, when the underlying SLOs are not met.

Should the computational resources of the system are not enough to uphold certain Service level objectives (SLOs), 
i.e. processing time for x amount frames, energy consumption, etc then the producer has some options to change required computational load. These include:

- **Quality:** Switch to a different grade YOLOv11 model.
- **FPS:** Change Source Stream FPS
- **Resolution:** Change source stream resolution

The goal is to maximize QoS by utilizing the available resources of distributed system (workers).

### Work-API
The Producer employs a `zeromq ROUTER` socket, that is waiting for requests and returning a task.
The communication is a two-part process:
1. The Worker sends a `REQ`
2. The Producer replies with a `REP`
#### Request-Reply Structure
`REQ:` A dict that defines the type of request and optional additional information.
```python
req = {
    type: str,
    #(optional additional information)
}
```

`REP:` multipart message containing
General information about the response:
```python
info = {
    type: str,
    #(optional additional information)
}
```
additionally for each task:
```python
task_metadata = {
    id: int,
    dtype: str,
    shape: tuple[int, int, int]
}
task: numpy.ndarray
```
##### Registration
TODO REST

## Worker
The workers constantly request work at the producer, processes it and forwards the result to the collector.

## Collector
The Collector implements a ``zeromq.PUSH`` socket that constantly accepts results from workers and re-orders them in order to form the final output video-stream

### Active Inference
In order to choose which measure is used to uphold QoS/SLOs, the edge node uses Active Inference.

## Producer
The producer creates the tasks and sends them to the workers upon receiving a request.

# Implementation
## Active Inference Model
- [pymdp](https://github.com/infer-actively/pymdp)
## Bayesian Inference Model (Alternative)
- [NumPyro](https://num.pyro.ai/en/stable/)
- [pyro](https://pyro.ai/)
- [TensorFlow Probability (TFP)](https://www.tensorflow.org/probability)
- [Stan](https://pystan.readthedocs.io/en/latest/)
- [PyMC](https://www.pymc.io/welcome.html)
- [InferPy](https://github.com/PGM-Lab/InferPy)
- [BayesPy](https://github.com/bayespy/bayespy)
## Which Library Should You Choose?
- For Active Inference: Use [pymdp](https://github.com/infer-actively/pymdp) if you want a dedicated library. For more flexibility, use [pyro](https://pyro.ai/) or [TensorFlow Probability (TFP)](https://www.tensorflow.org/probability) Probability.
- For Bayesian Modeling: Use [PyMC](https://www.pymc.io/welcome.html), [Stan](https://pystan.readthedocs.io/en/latest/), or [BayesPy](https://github.com/bayespy/bayespy).
- For High-Performance Inference: Use [NumPyro](https://num.pyro.ai/en/stable/) or [TensorFlow Probability (TFP)](https://www.tensorflow.org/probability).

DeepSeek Recommendation
1. Pyro 
2. TensorFlow Probability (TFP):
Similar to Pyro but integrates with TensorFlow. Suitable for large-scale distributed systems.
3. NumPyro:
Lightweight and fast, built on JAX. Ideal for high-performance inference.
4. PyMC:
User-friendly and well-documented. Great for Bayesian modeling but less flexible for Active Inference.
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
