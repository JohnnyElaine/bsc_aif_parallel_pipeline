# Description
This is a general distributed Parallel Pipeline that processes streaming related tasks, in edge computing cases.
The idea being that data stream is split up into smaller chunks, which are then distributed to multiple workers (edge computing devices) running in parallel. 
The results of each worker are then collected and reconstructed outputting the modified stram.

For evaluation purposes a video-stream was chosen as the stream source. The main computational task is running YOLOv11 inference said live video stream. 

But this general framework is designed to work for arbitrary data streams with any type of computation that can be parallelized. (e.g. video-,audio-,LIDAR-,RADAR-,network-traffic-,sensor-data-streams, etc)

Given the resources limited nature of edge computing, the system has to implement some elasticity measures to upholding a certain level of Quality of Experience (QoE) / Quality of Service (QoS) depending on the stream type.
Elasticity specifically meaning to ability to dynamically manipulate certain quality parameters of the datastream in order to control overall computational load on the worker-nodes. 
Because we are in an edge computing scenario, we can't simply add more worker nodes when computational demand is high. 
This means that if the output stream of the parallel pipeline is not up to predefined QoE standard, then the stream must reduce quality parameters of some kind in order to reduce overall computational load on the system.

The decision making process regarding the elasticity is implemented using an Active Inference (AIF) based Agent. 
Active Inference (AIF) is an emerging concept from neuroscience that describes how the brain continuously predicts and evaluates sensory information to model real-world processes.
The goal if the AIF agent is to maximize the quality metrics of the output-stream while providing a predefined level of QoE. 
This delicate balance is implemented using certain preference regarding the quality metrics of the stream and Service Level Objectives (SLOs) in order to keep resource usage within bounds.

The distributed parallel pipeline consists of 3 programs:
- Producer
- Worker
- Collector

The producer creates tasks that are then distributed to the workers. The [workers](#collector) run a processes the tasks. The results are then sent to the [collector](#collector), where they are reconstructed in order to form the output stream.

The rest of the description will assume that the data-stream is a video-stream and the computational task is YOLOv11 inference.

## Producer
The producer constantly creates a stream of data. This data (e.g. frames in our example) comes in the form of `Task` objects. 

A `Task` consists of 3 fields:
- `id` The `integer` id of the task
- `type` The `string` type of tasks. e.g. YOLOv11 inference
- `data` A `numpy.ndarray` containing a single frame of the video
 `numpy.ndarray` with a `id` and `type`. 

For example, when running inference on a video stream the producer would produce tasks such as:
```
Task:
    id = 10
    type = 'INFERNCE'
    data = np.ndarray
```

During the producers runtime it continuously creates new tasks and stores them in an internal buffer, called the task queue. The tasks remain in this queue until a worker requests work via the [Work-API](#work-api).
This implies that if the tasks are produced faster than they are consumed, then the task queue will grow indefinitely until the end of the runtime. In order to mitigate this problem the producers acts as a type of [controlling entity](#controlling-entity). 
Meaning that it can manipulate certain parameters to increase/decrease the overall required computational load in order to mitigate the concerns of a growing task queue.

When the producer has finished it signals all workers to shut down by sending a `Task` with `type=END`

### Elasticity
The producer also serves as a controlling entity regarding elasticity for the rest of the network, holding the information about the current state of the video stream and propagating changes when they occur.
The secondary goal of the producer is to ensure maximum Quality of Experience (QoE) under the tight resource constraints of an edge computing network.
Quality of Experience (QoE) in this context defines 3 user-centric parameters:

- **FPS:** FPS of the video stream
- **Resolution:** Resolution of the video stream.
- **Quality:** Grade of the YOLOv11 model used.

In order to balance the QoE with the high demand for computational resources, the producer implements certain [Service Level Objectives (SLOs)](#service-level-objectives-slos).

Under the constraints of the SLOs the producer aims to maximise the following 3 parameters (goals):
1. keep resolution as close as possible to the source-resolution of the video-stream
2. keep fps as close as possible to the source-fps of the underlying video-stream
3. maximize the result of the YOLOv11 inference (Maximize WorkLoad)

While the producer tries to fulfill the SLOs and maximize parameters (Workload, fps, resolution) at the same time, it is crucial that the SLO are of a much higher priority compared to the parameters.
Especially since the parameters directly influence the probability of fulfilling the SLOs. Maximizing the video stream parameters is more a preference, rather than a priority

#### Video Stream Parameters
The producers implements elasticity by directly controlling the 3 video stream parameters.

If the computational resources of the system are not enough to uphold certain Service level objectives (SLOs), then the AIF agent may reduce the video stream parameters by:

- **Quality:** Switch to a different grade YOLOv11 model.
- **FPS:** Change source stream FPS
- **Resolution:** Change source stream resolution

**Goal:** The goal is to maximize QoE (video stream parameters) by utilizing the available resources of distributed system (workers), while fulfilling the given Service Level Objectives (SLOs).

Keep in mind that higher video stream parameters lead to a higher computational demand from the workers.

### Service Level Objectives (SLOs)
A Service Level Objective (SLO) is a measureable objective that a system can enfoce. In this case we choose our SLOs in a way so that the producer can elastically adapt, by changing the 3 quality parameters (Quality, Resolution, FPS)

The Service Level Objectives (SLOs) are implemented by the Producer in order to ensure the highest possible Quality of Experience (QoE) given the current available resources.

Possible SLOs include: processing time for x amount frames, energy consumption, buffer size, memory usage, etc.

A SLO can have 3 types of states:
- **OK:** The SLO is fulfilled
- **WARNING:** The SLO is fulfilled, but close the the threshold of being unfulfilled
- **CRITICAL:** The SLO is unfulfilled

Currently only 2 SLOs are implemented:
- Memory Usage
- Queue Size

#### Memory Usage
```
memory_usage <= X%
```
- `X%` maximum percentage of acceptable memory use

GOAL: Ensure memory usage does not exceed capacity. Consuming the entire memory of the producer will cause a massive slowdown, as the new tasks will either be stored on a slower type of storage or discarded entirely.

#### Task queue (buffer) size
```
task_queue <= X
```
- ``X`` maximum acceptable number of tasks. e.g. ``X = source_fps * 2``

GOAL: Make sure there is enough compute power to handle tasks in real time. This SLO makes sure that the internal task buffer of the producer does not grow indefinitely. If the buffer remains below the chosen threshold, then this indicates that there is enough computational resources available on the worker side to handle the demand of the current video stream parameters.

## Worker
The workers purpose is to process the tasks provided by the producer. It continuously
1. request work from the producer.
2. processes it by some form of computation (e.g.  YOLOv11 inference)
3. sends the result to the collector.

### Requesting Work
The worker requests work whenever its internal task-buffer (a queue containing TODO tasks ) is empty. The work requesting is done according to the [Work-API](#work-api).

### Processing the task
The worker takes tasks from the task-buffer and runs inference on them using YOLOv11. The result is then stored in the result-buffer (a queue containing processed tasks).

The computation is done in a separate process to enable maximum performance.
### Sending Results to Collector
The worker takes the results (processed tasks) from the result-buffer and sends them to the collector using a zeromq.PUSH`

#### Results
Results are identical to ``Task`` objects, with the only difference being that `type=COLLECT`

These 3 concerns (requesting, processing, sending) are implemented in their own threads/processes in order to achieve a non-blocking task processing pipeline.

## Collector
The collector continuously accepts results from  
The Collector implements a ``zeromq.PULL`` socket that constantly accepts results from workers and reconstructs them to produce the final output video-stream.


## Work-API
Implemented by the Producer and used by the Worker.

The Producer implements a `zeromq.ROUTER` socket, that is waiting for requests and returning a task.
The Worker implements a `zeromq.REQ` socket, that is requesting tasks from the Producer's `zeromq.ROUTER` socket.

The communication is request-reply structure:
1. The Worker sends a `REQ`
2. The Producer replies with a `REP`

This is often referred to as the [Load Balancing Pattern](https://zguide.zeromq.org/docs/chapter3/#The-Load-Balancing-Pattern). 
This approach maximises the resource utilization of the workers, as each worker requests work up to its own maximum capacity. 
The downside of such an architecture is the added overhead, as the workers have to explicitly request work, whenever their internal buffer is empty.
This overhead is justified as it negligible in comparison to the large amounts of ``Task.data`` that sent via reply of the producer and enables proper load balancing of the worker nodes. 

### General Request-Reply Structure
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
```
```python
data: numpy.ndarray
```

So for example a response containg a singular ``Task`` would look like this:
A multipart message containg
```python
info, task_metadata
``` 
```python
{type: str, #(optional additional information)}, {id, dtype, shape}, task
``` 
```python
{type=INFERENCE}, {id=0, dtype=int64, shape=(1920,1080,3)}, np.ndarray
``` 

### Regular REQ-REP
##### Register -> Confirmation
When a worker starts it sends a registration message to the producer in order to get the current stream configuration.
This makes sure that the worker does not need to be configured individually, solely relying on the provided configuration of the producer.

The worker then loads the required model(s), depending on the retrieved configuration, and adjusts the internal configuration. Upon completing the setup process, the worker starts with the work requesting process
`REQ:` 
```python
req = {
    type='REGISTER'
}
```

`REP:` 
```python
info = {
    type='REGISTRATION_CONFIRMATION',
    work_type:str ∈ ['NONE', 'YOLO_DETECTION', 'YOLO_OBB']
    work_type:str ∈ ['LOW', 'MEDIUM', 'HIGH']
    loading_mode:str ∈ [0 (LAZY), 1 (EAGER)]
}
```
##### Request Work -> Receive Work
General mode of operation.

`REQ:` 
```python
req = {
    type='GET_WORK'
}
```

`REP:` multipart message containing
```python
info = {
    type:str ∈ ['INFERENCE', (Other potential work types)]
}
```
additionally for each task:
```python
task_metadata = {
    id: int,
    dtype: str,
    shape: tuple[int, int, int]
}
```
```python
data: numpy.ndarray
```

##### Request Work -> Receive Change
This happens when there is a crucial change in the current parameter configuration of the stream. The producer replies with a list of changes instead of tasks. 
This is because it is of the utmost importance that the workers adjust to the changes before continuing processing new tasks.

`REQ:` 
```python
req = {
    type='GET_WORK'
}
```

`REP:`
```python
info = {
    type='CHANGE'
    (optional)  CHANGE_WORK_LOAD:int ∈ [0 (LOW), 1(MEDIUM), 2 (HIGH)]
    (optional) change-2=value2
    (optional) change-3=value2
    ...
}
```


For example if the work load for the YOLOv11 inference needs to be changed to 1 (MEDIUM) a ``REP`` would look like this:

`REP:`
```python
info = {
    type='CHANGE'
    CHANGE_WORK_LOAD=1
}
```

### Other Responses
#### Any Rquest -> END of transmission
This occurs when the producer is stopping and has no more tasks left. This signals the workers and by extension the collector to shut down.
```python
req = {
    type=ANY
}
```

`REP:`
```python
info = {
    type='END'
}
```
TODO REST
# ---------------------------------------------------------------

# Dependencies
## How to build `requirements.txt`
```pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics
pip install msgpack
pip install pyzmq
pip install inferactively-pymdp
pip install stable-baselines3
pip install gym


pip freeze > requirements.txt
```

Add on top of requirements.txt
```
-f https://download.pytorch.org/whl/cu126/torch
-f https://download.pytorch.org/whl/cu126/torchvision
-f https://download.pytorch.org/whl/cu126/torchaudio
```

# TODO
AIF Agent impl mit intelligentVehicle vergleichen, Fix agent: somehow inverse, fps are constantly being changed
Finish README


Task Generation
    # TODO:
    # When FPS is lower then original video fps
    # Skip frames so video is still in real time, just with lower fps
    # currently reducing fps will increase the total streaming time
    # the total streaming time must stay consistent (length of original video)



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
