# Description
This python project implements a distributed parallel pipeline for processing streaming data in edge computing environments.
The main idea is to split a data stream into smaller chunks that are distributed across multiple workers (edge nodes) running in parallel. 
Each worker node processes its chunk, and the results are aggregated by a collector node to form a modified output stream.

The architecture draws inspiration from a [Apache Storm Topology](https://storm.apache.org/releases/2.7.1/Tutorial.html) and [Apache Flink](https://flink.apache.org/), consisting of:
A single Sprout (The collector), an internal layer of Bolts (Workers) and an aggregate Bolt (Collector).


For evaluation, a video stream of a highway is used as input, with YOLOv11 running inference in order to detect vehicles on live video frames. 
However, the framework is generic and applicable to any parallelizable data stream, such as audio, LIDAR, RADAR, network traffic, or sensor data.

**Central Research Question:**
"How can i efficiently process data streams in resource limited edge computing cases"
This is done by:
1. implementing a distributed pipeline for parallel processing of data.
2. employing Service Level Objectives (SLOs) in order uphold quality standards and abide by resource constraints
3. implementing an active inference (AIF) based agent in order to control elasticity demands

## Motivation
Edge computing offers scalable, low-latency processing closer to data sources, but is constrained by limited resources. 
This pipeline supports elasticity—the ability to dynamically adjust stream quality parameters in order to uphold a certain level of Quality of Experience (QoE) / Quality of Service (QoS) depending on the stream type, even when resources are constraint.
Elasticity specifically meaning to ability to dynamically manipulate certain quality parameters of the datastream in order to control overall computational load on the worker-nodes. 
Because we are in an edge computing scenario, we can't simply add more worker nodes when computational demand is high. 
This means that if the output stream of the parallel pipeline is not up to predefined QoE standard, then the stream must reduce quality parameters of some kind in order to reduce overall computational load on the system.

The decision making process regarding the elasticity is implemented using an Active Inference (AIF) based Agent. 
Active Inference (AIF) is an emerging concept from neuroscience that describes how the brain continuously predicts and evaluates sensory information to model real-world processes.
The goal if the AIF agent is to maximize the quality metrics of the output-stream while providing a predefined level of QoE. 
This delicate balance is implemented using certain preference regarding the quality metrics of the stream and Service Level Objectives (SLOs) in order to keep resource usage within bounds.

## Architecture
The pipeline consists of three main components:

- Producer: Generates and controls task flow.
- Worker: Processes tasks (e.g., YOLO inference).
- Collector: Aggregates processed results.

Throughout this README, examples refer to video streams and YOLOv11 inference, but the pipeline is general-purpose.
## Producer
The Producer continuously generates a stream of tasks. A ``Task`` object represents a single unit of work (e.g. a video frame).

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

During the producers runtime it continuously creates new tasks and stores them in an internal buffer, called the task queue. The tasks remain in this queue until a worker requests work via the [Work-API](#work-api), consuming the task.
If the tasks are produced faster than they are consumed, then the task queue will grow indefinitely. In order to mitigate this problem the producers acts as a type of [controlling entity](#controlling-entity). 
Meaning that it can manipulate certain parameters to increase/decrease the overall required computational load in order to mitigate the concerns of a growing task queue.

When the producer has finished it signals all workers to shut down by sending a `Task` with `type=END`

### Elasticity
The producer also monitors and adjusts stream parameters to maintain user-defined QoE under constrained resources
It tracks information about the current state of the video stream and propagating changes when they occur.
The three QoE parameters are:

1. **FPS:** FPS of the video stream
2. **Resolution:** Resolution of the video stream.
3. **Quality:** Grade of the YOLOv11 model used.

It aims to:
- Maximize these parameters
- While meeting certain [Service Level Objectives (SLOs)](#service-level-objectives-slos): Memory Usage, Task Queue Size

Under the constraints of the SLOs the producer aims to maximise the following 3 parameters (goals):
1. keep resolution as close as possible to the source-resolution of the video-stream
2. keep fps as close as possible to the source-fps of the underlying video-stream
3. maximize the result of the YOLOv11 inference (Maximize Inference Quality)
Keep in mind that higher stream parameters lead to a higher computational demand from the workers.

#### Video Stream Parameters
**Stream Parameter Adjustment:** If SLOs are at risk, the producer (via the AIF agent) may:

- **Quality:** Switch to a different grade YOLOv11 model.
- **FPS:** Change source stream FPS
- **Resolution:** Change source stream resolution

While the producer tries to fulfill the SLOs and maximize parameters (inference quality, fps, resolution) at the same time, it is crucial that the SLO are of a much higher priority compared to the parameters.
Especially since the parameters directly influence the probability of fulfilling the SLOs. Maximizing the video stream parameters is more a preference, rather than a priority

#### Changing of streaming parameters
The producer is able to directly control the streaming parameters. In this example these are Resolution, FPS, Quality

### Service Level Objectives (SLOs)
A Service Level Objective (SLO) is a measureable objective that a system can enforce. In this case we choose our SLOs in a way so that the producer can elastically adapt, by changing the 3 quality parameters (Quality, Resolution, FPS)
The Service Level Objectives (SLOs) are implemented by the Producer in order to ensure the highest possible Quality of Experience (QoE) given the current available resources.
Possible SLOs include: processing time for x amount frames, energy consumption, buffer size, memory usage, etc.

SLOs guide the system to maintain QoE within operational limits. Each SLO can be in one of three states:
- **OK:** The SLO is fulfilled
- **WARNING:** The SLO is fulfilled, but close the the threshold of being unfulfilled
- **CRITICAL:** The SLO is unfulfilled

Currently only 2 SLOs are implemented:
- Memory Usage
- Queue Size

#### Representation in the code
The SLOs are calculated as float values with bound 0.0-infinity. The higher the value of the SLO, the worse its status. 
- ``0.0 <= slo-value < 0.9`` indicates a fulfilled SLO in a good state.
- ``0.9 <= slo-value < 1.0`` indicates a fulfilled SLO in a warning state.
- ``1.0 <= slo-value < MAX_FLOAT`` indicates an unfulfilled SLO in a critical state

#### Memory Usage SLO
```
memory_usage <= X%
```
- `X%` maximum percentage of acceptable memory use

GOAL: Ensure memory usage does not exceed capacity. Consuming the entire memory of the producer will cause a massive slowdown, as the new tasks will either be stored on a slower type of storage or discarded entirely.

#### Task queue (buffer) size SLO
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
The Collector implements a ``zeromq.PULL`` socket that constantly accepts results from workers and aggregates them to produce the final output video-stream.


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
    (optional)  CHANGE_INFERENCE_QUALITY:int ∈ [0 (LOW), 1(MEDIUM), 2 (HIGH)]
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
    CHANGE_INFERENCE_QUALITY=1
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

## Implementation of Active Inference
The agent that controls the elasticity of the system runs on the producer. It is implemented using the active inference library [pymdp](https://github.com/infer-actively/pymdp). More information is available in the offical [paper](https://arxiv.org/abs/2201.03904).
In order to make intelligent decisions using active inference, we need to model the problem as a Partially observable Markov decision process (POMDP). For this the environment was modeled as follows:

Observations:
- FPS of stream
- Resolution of stream
- YOLOv11 Inference Quality (Model) used by the worker nodes
- State of the Memory SLO (as a float value)
- State of the Queue Size SLO (as a float value)

Actions:
- Changing of the FPS of the stream
- Changing the Resolution of the stream
- Changing the YOLOv1 Inference Quality (The model used by the worker nodes)

### AIF Loop
The agent defines an interval for the AIF loop. Per default this is 1 second.
Meaning every 1s:
- The agent retrieves the observations (SLO values and stream quality parameters)
- The agent updates its believes
- The agent chooses actions according to its believes

## Evaluation
In order to evaluate the effectiveness of the implementation we implement the following:

### Additional Agents
We implement other types of agents that control the elasticity of the system. We implement two different agent types:
- Active Inference Agent (AIF), The one described above, i.e. the main one of this paper
- Heuristic Agent
- Reinforcement Learning (RL) Agent

All agents share the same goal of upholding the SLOs while trying to keep high QoE, through elasticity (Changing of quality parameters).

#### Reinforcement Learning Agent
This agent serves as a direct comparison to the active inference agent. This is because Reinforcement Learning (RL) is also framework for modeling intelligent behavior, but with a different fundamental underlying principle.
The agent is implemented using [stable-baselines](https://github.com/Stable-Baselines-Team/stable-baselines) and [Gymnasium (gym)](https://github.com/Farama-Foundation/Gymnasium).

#### Heuristic Agent
In order to evaluate the effectiveness of statistical models, such as AIF or RL, we implemented a simple agent based on heuristic measurements.

The agent defines an interval for the loop. Per default this is 1 second.
Meaning every 1s:
- The agent retrieves the values of the SLO
- The agent evaluates the values.
- If a SLO is in CRITICAL state (i.e. value > 1.0) the agent reduces the stream quality parameters
- If a SLO is in WARNING state, the agent does nothing.
- If all SLOs are in OK state, the agent tries to improve the stream quality parameters.

### Simulation
In order to properly evaluate the agent types we have set a simulation. The simulation consists of a simple streaming scenario.

#### Environment
The entire simulation is running a single computer with the specs:
- **Operating System:** Windows 11, Version 23H2 (Build 22631.5335)
- **Python:** Python 3.12.2
- **CPU:** AMD Ryzen 7 7800X3D
- **GPU:** Nvidia GeForce GTX 1660Ti (MSI GTX Ti Ventus XS OC)
- **Memory:** 32GB DDR5 RAM, Clock: 4800MHz, Dual Channel, Timing: 40-40-40-77, tRC: 117, tRFC: 708
- **Drive:** WD Black SN770 2TB

#### Setup
- 1 Producer
- n Workers (3 Workers typically)
- 1 Collector

The video stream generated by the producer is a video of highway traffic. The goal is to detect the vehicles passing by using YOLOv11 and drawing bounding boxes around them.
All workers have the same amount of processing power.  

In order to measure the elastic capabilities of the agents, m number (1 typically) of the agents will go offline temporarily causing the total amount of computational resources of the distributed system to decrease.
This will cause the values of the SLOs to change into a critical state. This will prompt to agent to take an action in order to achieve equilibrium.
This will be done by decreasing the quality parameters in some way, decreasing the computational requirements in such a way that the remaining workers are able to uphold the SLO.
After a certain period the agents will recover and come back online. This will cause the total total amount of computational resources of the distributed system to increase.
The agent should then notice this increasing in available resources and try to increase the stream quality parameters to facilitate a better QoE.
The outage and recovery of the workers are set at 25% and 75% of the total stream duration respectively.

During the entire runtime of the simulation the producer tracks certain metrics:
- Registration time of workers
- Number of requested tasks per worker
- Stream Quality Parameters Capacity (FPS, Resolution, Inference Quality) over time. The capacity is represented as a float value (0.0-1.0) Where 0 is the worst possible configuration and 1.0 is the best possible configuration.
- Queue Size over time (Number of Tasks in queue)
- Memory Usage % over time
- Queue Size SLO value over time
- Memory Usage SLO value over time
and return as a python dataframes.

These values are then plotted using [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/). 


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
