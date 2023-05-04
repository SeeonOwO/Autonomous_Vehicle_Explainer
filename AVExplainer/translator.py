import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from DataLoader import *
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from EvidentialNet import *
import cv2
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from torchvision import transforms

action_labels = ['move forward', 'stop/slow down', 'turn left', 'turn right']
reason_labels = [
    'Forward - follow traffic',
    'Forward - the road is clear',
    'Forward - the traffic light is green',
    'Stop/slow down - obstacle: car',
    'Stop/slow down - obstacle: person/pedestrian',
    'Stop/slow down - obstacle: rider',
    'Stop/slow down - obstacle: others',
    'Stop/slow down - the traffic light',
    'Stop/slow down - the traffic sign',
    'Turn left - front car turning left',
    'Turn left - on the left-turn lane',
    'Turn left - traffic light allows',
    'Turn right - front car turning right',
    'Turn right - on the right-turn lane',
    'Turn right - traffic light allows',
    "Can't turn left - obstacles on the left lane",
    "Can't turn left - no lane on the left",
    "Can't turn left - solid line on the left",
    "Can't turn right - obstacles on the right lane",
    "Can't turn right - no lane on the right",
    "Can't turn right - solid line on the left"
]

def generate_explanation(action_idx, reason_idx):
    action = action_labels[action_idx]
    reason = reason_labels[reason_idx]
    explanation = f"{action.capitalize()}! {reason}"
    return explanation


cfg = get_cfg()
img = cv2.imread('test.png')
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)
predictions = predictor(img)
ins = predictions["instances"]
print(ins.pred_classes)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"
model = build_model(cfg)
model.eval()
if len(cfg.DATASETS.TEST):
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

x = cv2.imread("test.png")
import detectron2.data.transforms as T
aug = T.ResizeShortestEdge([704, 704], 1280)
with torch.no_grad():
    height, width = x.shape[:2]
    y = aug.get_transform(x).apply_image(x)
    y = torch.as_tensor(y.astype("float32").transpose(2, 0, 1))
    print(y.shape)
    inputs = {"image": y, "height": height, "width": width}
    outputs = model([inputs])
    print(outputs[0]["instances"].pred_classes)
from detectron2.structures import ImageList
with torch.no_grad():
    height, width = x.shape[:2]
    # z = aug.get_transform(x).apply_image(x)
    z = cv2.resize(x, (1280,736))
    z = (z - cfg.MODEL.PIXEL_MEAN) / cfg.MODEL.PIXEL_STD
    resized_image = z.astype("float32").transpose(2, 0, 1)
    resized_image_tensor = torch.as_tensor(resized_image).unsqueeze(0).to("cuda")
    print(cfg.MODEL.PIXEL_MEAN)
    print(cfg.MODEL.PIXEL_STD)
    image_list = ImageList(resized_image_tensor, [(resized_image_tensor.shape[-2], resized_image_tensor.shape[-1])])
    # Split the model
    backbone = model.backbone
    proposal_generator = model.proposal_generator
    roi_heads = model.roi_heads

    # Run the model components
    features = backbone(resized_image_tensor)

    proposals, _ = proposal_generator(image_list, features)
    instances, _ = roi_heads(resized_image_tensor, features, proposals)

    print(instances[0].pred_classes)