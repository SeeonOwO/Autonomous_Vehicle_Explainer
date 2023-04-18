import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

import detectron2.modeling
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures.image_list import ImageList

import os
import json
import cv2


class CarExplanationDataset(Dataset):
    def __init__(self, data_path, label_file, transform=None):
        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = list(self.labels.keys())[idx]
        img_path = os.path.join(self.data_path, image_name + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert the image to a PIL image

        if self.transform:
            img = self.transform(img)

        actions = self.labels[image_name]['actions']
        reasons = self.labels[image_name]['reason']

        return img, torch.tensor(actions), torch.tensor(reasons)


class CarActionModel(nn.Module):
    def __init__(self, cnn_backbone, num_actions, num_reasons):
        super(CarActionModel, self).__init__()

        # Initialize the Faster R-CNN feature extractor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.frcnn = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.frcnn)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.cnn_backbone = cnn_backbone
        self.action_head = nn.Sequential(
            nn.Linear(2048, num_actions),
            nn.LogSoftmax(dim=1)
        )
        self.reason_head = nn.Sequential(
            nn.Linear(2048, num_reasons),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Extract global feature
        global_features = self.cnn_backbone(x)


        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        all_local_features = []

        
        for img in x:
            img = img.cpu().numpy().transpose(1, 2, 0)
            outputs = predictor(img)
            boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
            scores = outputs['instances'].scores.cpu().numpy()
            sorted_indices = np.argsort(scores)[::-1]
            boxes = boxes[sorted_indices]
            boxes = boxes[:5]
            cropped_imgs = [img[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in boxes]

            with torch.no_grad():
                local_features = []
                if len(cropped_imgs) > 0:
                    for cropped_img in cropped_imgs:
                        cropped_img = cropped_img.astype('float32') / 255.0
                        local_feature = self.cnn_backbone(
                            torch.tensor(cropped_img).unsqueeze(0).permute(0, 3, 1, 2).to(x.device))
                        local_features.append(local_feature)
                else:
                    local_features = [torch.zeros_like(global_features)]

                local_features = torch.stack(local_features, dim=0).mean(dim=0, keepdim=True)
                all_local_features.append(local_features)

        all_local_features = torch.cat(all_local_features, dim=0)
        merged_features = torch.cat((global_features, all_local_features), dim=1)

        actions = self.action_head(global_features)
        reasons = self.reason_head(global_features)

        return actions, reasons

    def _extract_features_and_proposals(self, images):
        # images = detectron2.modeling.preprocess_image(images)
        features = self.frcnn.backbone(images)
        proposals, _ = self.frcnn.proposal_generator(images, features)
        return features, proposals


def get_resnet50_backbone():
    model = resnet50(weights="ResNet50_Weights.DEFAULT")
    model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    return model


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, actions, reasons in train_loader:
        images = images.to(device)
        actions = actions.float().to(device)
        reasons = reasons.float().to(device)

        optimizer.zero_grad()

        pred_actions, pred_reasons = model(images)
        loss = criterion(pred_actions, actions) + criterion(pred_reasons, reasons)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, actions, reasons in val_loader:
            images = images.to(device)
            actions = actions.to(device)
            reasons = reasons.to(device)

            pred_actions, pred_reasons = model(images)
            loss = criterion(pred_actions, actions) + criterion(pred_reasons, reasons)

            running_loss += loss.item()

    return running_loss / len(val_loader)


def generate_explanation(action_idx, reason_idx):
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
    action = action_labels[action_idx]
    reason = reason_labels[reason_idx]
    explanation = f"{action.capitalize()}! {reason}"
    return explanation


def test(model, dataloader, device):
    model.eval()
    correct_actions = 0
    correct_reasons = 0
    total = 0

    with torch.no_grad():
        for inputs, actions, reasons in dataloader:
            inputs, actions, reasons = inputs.to(device), actions.to(device), reasons.to(device)

            action_preds, reason_preds = model(inputs)
            print(action_preds)
            print(reason_preds)
            _, action_preds = torch.max(action_preds, 1)
            _, reason_preds = torch.max(reason_preds, 1)

            total += actions.size(0)
            correct_actions += (action_preds == actions).sum().item()
            correct_reasons += (reason_preds == reasons).sum().item()

            # Print explanations for the first few samples in the test set
            if total <= 5:
                for i in range(inputs.size(0)):
                    action_idx = action_preds[i].item()
                    reason_idx = reason_preds[i].item()
                    explanation = generate_explanation(action_idx, reason_idx)
                    print(f"Image {total - inputs.size(0) + i + 1}: {explanation}")

    action_accuracy = 100 * correct_actions / total
    reason_accuracy = 100 * correct_reasons / total

    print(f"Test Action Accuracy: {action_accuracy:.2f}%")
    print(f"Test Reason Accuracy: {reason_accuracy:.2f}%")


def main():
    cnn_backbone = get_resnet50_backbone()

    # Define the paths to your dataset and annotations
    train_img_dir = "C:/Users/lsion/Desktop/lastframe/train"
    val_img_dir = "C:/Users/lsion/Desktop/lastframe/validate"
    test_img_dir = "C:/Users/lsion/Desktop/lastframe/test"

    train_annotation_path = "C:/Users/lsion/Desktop/lastframe/gt_4a_21r_train.json"
    val_annotation_path = "C:/Users/lsion/Desktop/lastframe/gt_4a_21r_val.json"
    test_annotation_path = "C:/Users/lsion/Desktop/lastframe/gt_4a_21r_test.json"

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset instances for train, validation, and test sets
    train_dataset = CarExplanationDataset(train_img_dir, train_annotation_path, transform=transform)
    val_dataset = CarExplanationDataset(val_img_dir, val_annotation_path, transform=transform)
    test_dataset = CarExplanationDataset(test_img_dir, test_annotation_path, transform=transform)

    # Create DataLoader instances for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_actions = 4
    num_reasons = 21
    model = CarActionModel(cnn_backbone, num_actions, num_reasons)
    learning_rate = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(50):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        print(f"Epoch {epoch + 1}/{50}")
        print(f"Train Loss: {train_loss:.4f}")
    test(model, test_loader, device)


if __name__ == '__main__':
    main()
