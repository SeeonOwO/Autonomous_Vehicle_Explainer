import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from DataLoader import *
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler


class ExplanationModel(nn.Module):
    def __init__(self, detectron_model, lstm_hidden_size, num_actions, num_reasons):
        super(ExplanationModel, self).__init__()
        self.detectron_model = detectron_model
        self.lstm_hidden_size = lstm_hidden_size
        self.num_actions = num_actions
        self.num_reasons = num_reasons

        # Use the Fast R-CNN model's backbone for global feature extraction
        self.backbone = detectron_model.backbone

        self.rpn = detectron_model.proposal_generator

        # Use the Fast R-CNN model's ROI head for local feature extraction
        self.roi_heads = detectron_model.roi_heads

        self.fc = nn.Linear(512 * 7 * 7, 2048)

        # Define the LSTM model
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden_size, batch_first=True)

        # Define the output layers for actions and reasons
        self.action_output = nn.Linear(lstm_hidden_size, num_actions)
        self.reason_output = nn.Linear(lstm_hidden_size, num_reasons)

    def forward(self, x):
        # print(x.size())

        # Global feature extraction
        raw_features = self.backbone(x)

        # Get the global feature from the last layer (e.g., 'p5')
        global_features = raw_features['p5']
        global_features_downsampled = F.adaptive_avg_pool2d(global_features, output_size=(7, 7))
        # print(global_features_downsampled.size())

        # Get the proposals
        image_list = ImageList(x, [(x.shape[-2], x.shape[-1])] * x.shape[0])
        proposals, _ = self.rpn(image_list, raw_features)
        instances, _ = self.roi_heads(x, raw_features, proposals)

        output_size = (7, 7)
        scales = (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = 0
        roi_pool = ROIPooler(output_size, scales, sampling_ratio, pooler_type="ROIAlign")
        boxes_per_image = [instances_i.pred_boxes for instances_i in instances]
        feature_map = [raw_features[f] for f in self.roi_heads.in_features]
        object_features = roi_pool(feature_map, boxes_per_image)  # Notice that boxes is enclosed in a list
        # print(object_features.size())

        batch_size = x.size(0)
        max_num_boxes = max([len(boxes) for boxes in boxes_per_image])
        weights = torch.ones(batch_size, max_num_boxes, device=object_features.device) / max_num_boxes
        weighted_sum = torch.bmm(weights.unsqueeze(1), object_features.view(batch_size, max_num_boxes, -1)).squeeze(1)
        weighted_sum_reshaped = weighted_sum.view(batch_size, 256, 7, 7)
        combined_features = torch.cat([global_features_downsampled, weighted_sum_reshaped], dim=1)
        # print(combined_features.size())

        # Reshape global_features tensor to 2D: (batch_size, channels * height * width)
        combined_features_reshaped = combined_features.view(combined_features.size(0), -1)

        # Pass reshaped global features through the fully connected layer
        combined_features_reduced = self.fc(combined_features_reshaped)

        # Pass reduced global features through LSTM
        lstm_out, _ = self.lstm(combined_features_reduced.unsqueeze(0))

        lstm_out = lstm_out.squeeze(0)

        # Generate action and reason outputs
        actions = self.action_output(lstm_out)
        reasons = self.reason_output(lstm_out)

        return actions, reasons

    '''
        # Create ImageList object
        image_sizes = [(int(x.size(2) / s), int(x.size(3) / s)) for s in [4, 8, 16, 32, 64]]
        images = ImageList(x, image_sizes)

        # Generate proposals
        proposals, _ = self.detectron_model.proposal_generator(images, global_features)
        proposals = [proposal[:1000] for proposal in proposals]  # Limit the number of proposals to avoid shape mismatch

        # Local feature extraction
        box_features_list = []
        local_features_list = []
        for proposal in proposals:
            box_features = self.roi_heads._shared_roi_transform(
                [global_features[f] for f in self.roi_heads.in_features], [proposal]
            )
            local_features = self.roi_heads.box_head(box_features)
            box_features_list.append(box_features)
            local_features_list.append(local_features)

        # Combine global and local features
        combined_features_list = [
            torch.cat((global_features['p5'][i], local_features), dim=1)
            for i, local_features in enumerate(local_features_list)
        ]

        # Pass combined features through LSTM
        lstm_out_list = []
        for combined_features in combined_features_list:
            lstm_out, _ = self.lstm(combined_features.unsqueeze(0))
            lstm_out_list.append(lstm_out)

        # Generate action and reason outputs
        actions = torch.stack([self.action_output(lstm_out[:, -1, :]) for lstm_out in lstm_out_list])
        reasons = torch.stack([self.reason_output(lstm_out[:, -1, :]) for lstm_out in lstm_out_list])

        return actions, reasons
    '''


'''
def main():
    # Your existing code for setting up the data loaders and iterating through the train_loader
    transform = transforms.Compose([
        transforms.Resize((736, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_train.json', 'r') as f:
        train_annotations = json.load(f)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_val.json', 'r') as f:
        val_annotations = json.load(f)

    train_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/train', train_annotations, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    model = build_model(cfg)
    lstm_hidden_size = 512
    num_actions = 4
    num_reasons = 21
    explanation_model = ExplanationModel(model, lstm_hidden_size, num_actions, num_reasons)
    device = torch.device("cuda")
    explanation_model.to(device)
    explanation_model.eval()
    inputs, actions_gt, reasons_gt = next(iter(train_loader))
    inputs = inputs.to(device)
    actions_gt = actions_gt.to(device)
    reasons_gt = reasons_gt.to(device)
    with torch.no_grad():
        actions_pred, reasons_pred = explanation_model(inputs)
    print(actions_pred)
    print(reasons_pred)
    print(actions_gt)
    print(reasons_gt)


if __name__ == '__main__':
    main()
'''