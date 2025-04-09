# Dependencies
## How to build `requirements.txt`
```pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install msgpack
pip install pyzmq
pip install inferactively-pymdp
pip install stable-baselines3


pip freeze > requirements.txt
```

Add on top of requirements.txt
```
-f https://download.pytorch.org/whl/cu124/torch
-f https://download.pytorch.org/whl/cu124/torchvision
-f https://download.pytorch.org/whl/cu124/torchaudio
```

# Description
This is a distributed Parallel Pipeline using active inference to dynamically change the size/frequency/complexity of the tasks, based on the current available computational resources of the system.
The goal is to run YOLOv11 inference on a live video stream using multiple distributed workers in parallel.
It consists of 3 program types:

- Producer
- Worker
- Collector

## Producer
The producer constantly creates work (frames of the source video stream) in the form of multiple `Task` objects. A `Task` is `numpy.ndarray` with a `id` and `type`. 
For example, when running inference on a video stream the producer would produce tasks such as:
```
Task:
    id = 10
    type = 'INFERNCE'
    data = np.ndarray
```
The produced tasks are then buffered and ready to be requested by a worker via the [Work-API](#work-api).

### Controlling Entity
Using the [Work-API](#work-api) the producer also serves as a controlling entity for the rest of the network holding the information about the current state of the video stream and propagating changes when they occur.
The secondary goal of the producer is to ensure maximum Quality of Experience (QoE). 

This means the producer tries to fulfill the following [Service Level Objectives (SLOs)](#service-level-objectives-slos)

Under the constraints of the SLOs the  producer aims to maximise the following parameters (goals) in the following order:
1. keep resolution as close as possible to the source-resolution of the video-stream
2. keep fps as close as possible to the source-fps of the underlying video-stream
3. maximize the result of the YOLOv11 inference (Maximize WorkLoad)

While the producer tries to fulfill the SLOs and maximize parameters (Workload, fps, resolution) at the same time, it is crucial that the SLO are of a much higher priority compared to the parameters.
Especially since the parameters directly influence the probability of fulfilling the SLOs. Maximizing the parameters is more a preference, rather than a priority


## Worker
The workers constantly
1. request work at the producer.
2. processes it by running YOLOv11 inference.
3. sends the result to the collector. 

### Requesting Work
The Worker requests work whenever its internal task-buffer is empty. The work requesting is done according to the [Work-API](#work-api).

### Processing the task (image)
The Worker takes tasks (frames) from the task-buffer and runs inference on them using YOLOv11. The result is then stored in the result-buffer.

This is done in a separate process to enable maximum performance.
### Sending Results to Collector
The Worker implements a `zeromq.PUSH` socket that is used to immediately send all buffered results to the collector.

## Collector
The Collector implements a ``zeromq.PULL`` socket that constantly accepts results from workers and re-orders them to produce the final output video-stream.

## Elasticity
The Producer tries to ensure Quality of Service (QoS) by providing certain elasticity features, when the underlying SLOs are not met.

Should the computational resources of the system are not enough to uphold certain Service level objectives (SLOs), 
i.e. processing time for x amount frames, energy consumption, buffer size, memory usage, etc then the producer has some options to change required computational load. These include:

- **Quality:** Switch to a different grade YOLOv11 model.
- **FPS:** Change Source Stream FPS
- **Resolution:** Change source stream resolution

The goal is to maximize QoS by utilizing the available resources of distributed system (workers).

## Service Level Objectives (SLOs)
The Service Level Objectives (SLOs) are implemented by the Producer in order to ensure the highest possible Quality of Experience (QoE) given the current available resources.

A SLO can have 3 types of states:
- **OK:** The SLO is fulfilled
- **WARNING:** The SLO is fulfilled, but close the the threshold of being unfulfilled
- **CRITICAL:** The SLO is unfulfilled

### Memory Usage
```
memory_usage <= X%
```
- `X%` maximum percentage of acceptable memory use

GOAL: Ensure memory usage does not exceed capacity.

### Task queue size
```
task_queue <= X
```
- ``X`` maximum acceptable number of tasks. e.g. ``X = current_fps * 2``

GOAL: Make sure there is enough compute power to handle tasks in real time.

### Quality/Workload

GOAL: Workers should run YOLO-inference at higher quality if possible.

### Resolution
```
current_res >= source_res * T
```
- `T` Tolerance. e.g. `T = 0.9` for 90% Tolerance

GOAL: Video should run at source resolution if possible.



## Work-API
Implemented by the Producer and used by the Worker.

The Producer implements a `zeromq.ROUTER` socket, that is waiting for requests and returning a task.
The Worker implements a `zeromq.REQ` socket, that is requesting tasks from the Producer's `zeromq.ROUTER` socket.

The communication is request-reply structure:
1. The Worker sends a `REQ`
2. The Producer replies with a `REP`

### Load Balancing
This is often referred to as the [Load Balancing Pattern](https://zguide.zeromq.org/docs/chapter3/#The-Load-Balancing-Pattern). 
This approach maximises the resource utilization of the workers.

#### General Request-Reply Structure
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

# ---------------------------------------------------------------
# TODO
Check if Instruction dataclass is actually useful, or if it can be merged with Task dataclass
Document and test heuristic agent
Finish simulation

Properly close all programs after task generation is done. request_handler. find alternate stopping condition, if not all workers are online

Optional:
    When changing fps: notify the collector that the fps are changed
    Send multiple frames at once Producer -> Worker && Worker -> Collector


## How to send numpy arrays using 0MQ efficiently:
https://pyzmq.readthedocs.io/en/latest/howto/serialization.html#example-numpy-arrays


# Implementation
## Active Inference Model
- [pymdp](https://github.com/infer-actively/pymdp)
## Causel Inference via DAG
- [pgmpy](https://pgmpy.org/) used in in [intelligentVehicle](https://github.com/borissedlak/intelligentVehicle) 
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
- For Bayesian Modeling: Use [PyMC](hmttps://www.pymc.io/welcome.htl), [Stan](https://pystan.readthedocs.io/en/latest/), or [BayesPy](https://github.com/bayespy/bayespy).
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
