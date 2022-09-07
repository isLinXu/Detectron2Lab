import torch

from detectron2 import model_zoo

model = model_zoo.get("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml", trained=True)
print('model:', model)
print(model.state_dict().keys())
torch.save(model.state_dict(), "../data/model_parameter.pkl")


