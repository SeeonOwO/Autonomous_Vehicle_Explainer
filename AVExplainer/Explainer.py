import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from DataLoader import *
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from EvidentialNet import *


class ExplanationModel(nn.Module):
    def __init__(self, detectron_model, num_actions, num_reasons, use_edl=False):
        super(ExplanationModel, self).__init__()
        self.detectron_model = detectron_model
        self.num_actions = num_actions
        self.num_reasons = num_reasons
        self.use_edl = use_edl

        # Use the Fast R-CNN model's backbone for global feature extraction
        self.backbone = detectron_model.backbone

        self.backbone_before_fpn = self.backbone.bottom_up

        self.rpn = detectron_model.proposal_generator

        # Use the Fast R-CNN model's ROI head for local feature extraction
        self.roi_heads = detectron_model.roi_heads

        self.selector = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.fc = nn.Linear(2304 * 7 * 7, 2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.action_fc = nn.Linear(2304, num_actions)
        self.reason_fc = nn.Linear(2304, num_reasons)
        self.action_fc_edl = nn.Linear(2304, num_actions*2)
        self.reason_fc_edl = nn.Linear(2304, num_reasons*2)

    def forward(self, x):
        # Global feature extraction
        raw_features = self.backbone(x)

        # Get the global feature from the last layer (e.g., 'p5')
        global_features = self.backbone_before_fpn(x)['res5']
        global_features_downsampled = F.adaptive_avg_pool2d(global_features, output_size=(7, 7))

        # Get the proposals
        image_list = ImageList(x, [(x.shape[-2], x.shape[-1])] * x.shape[0])
        proposals, _ = self.rpn(image_list, raw_features)
        instances, _ = self.roi_heads(x, raw_features, proposals)

        # Get the object feature
        output_size = (7, 7)
        scales = (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = 0
        roi_pool = ROIPooler(output_size, scales, sampling_ratio, pooler_type="ROIAlign")
        boxes_per_image = [instances_i.pred_boxes for instances_i in instances]
        feature_map = [raw_features[f] for f in self.roi_heads.in_features]
        object_features = roi_pool(feature_map, boxes_per_image)  # Notice that boxes is enclosed in a list

        # Separate object features for each image
        object_features_per_image = []
        start_idx = 0
        for boxes in boxes_per_image:
            num_boxes = len(boxes)
            object_features_per_image.append(object_features[start_idx:start_idx + num_boxes])
            start_idx += num_boxes

        # Add global features to the local features of each object
        combined_features_per_image = []
        for i, object_features_i in enumerate(object_features_per_image):
            global_features_i = global_features_downsampled[i].unsqueeze(0)  # Add batch dimension
            combined_features_i = torch.cat(
                [object_features_i, global_features_i.expand(object_features_i.size(0), -1, -1, -1)], dim=1)
            combined_features_per_image.append(combined_features_i)

        attention_weights_per_image = []
        for combined_features_i in combined_features_per_image:
            selector_output = self.selector(combined_features_i)
            pooled_output = torch.mean(selector_output, dim=(2, 3))
            scores = F.softmax(pooled_output.squeeze(1), dim=0)
            attention_weights_per_image.append(scores)

        # Compute the weighted average of object features based on attention weights
        weighted_features_per_image = []
        for i, combined_features_i in enumerate(combined_features_per_image):
            attention_weights_i = attention_weights_per_image[i].unsqueeze(1).unsqueeze(2).unsqueeze(
                3)  # Shape: (num_objects, 1, 1, 1)
            weighted_features_i = attention_weights_i * combined_features_i  # Element-wise multiplication
            weighted_average_i = torch.sum(weighted_features_i, dim=0, keepdim=True)  # Shape: (1, 2304, 7, 7)
            weighted_features_per_image.append(weighted_average_i)

        final_feature = torch.cat(weighted_features_per_image, dim=0)  # Shape: (batch_size, 2304, 7, 7)

        final_feature = self.global_avg_pool(final_feature)

        flattened_feature_maps = final_feature.view(final_feature.shape[0], -1)

        if self.use_edl:
            actions = self.action_fc_edl(flattened_feature_maps).view(x.shape[0], self.num_actions, 2)
            reasons = self.reason_fc_edl(flattened_feature_maps).view(x.shape[0], self.num_reasons, 2)
        else:
            actions = self.action_fc(flattened_feature_maps)
            reasons = self.reason_fc(flattened_feature_maps)

        return actions, reasons
