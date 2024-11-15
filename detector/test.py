import torch
from ultralytics import YOLO


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Run inference with the YOLO11n model on the 'bus.jpg' image
#results = model("./img/istockphoto-522564791-612x612.jpg", device=device, show=True)
#results = model("./vid/4K Video of Highway Traffic! [KBsqQez-O4w].mp4", device=device, show=True, )
#results = model.track(source=0, device=device, tracker='./trackers/bytetrack.yaml', show=True)
