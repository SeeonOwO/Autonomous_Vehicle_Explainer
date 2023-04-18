from Explainer import *
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score


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
    val_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/validate', val_annotations, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

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

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create an optimizer
    optimizer = optim.Adam(explanation_model.parameters(), lr=0.001)

    # Determine the number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        explanation_model.train()
        explanation_model.backbone.eval()
        explanation_model.rpn.eval()
        explanation_model.roi_heads.eval()
        for param in explanation_model.backbone.parameters():
            param.requires_grad = False

        for param in explanation_model.rpn.parameters():
            param.requires_grad = False

        for param in explanation_model.roi_heads.parameters():
            param.requires_grad = False
        train_running_loss = 0.0

        '''
        running_loss = 0.0
        running_corrects_actions = 0
        running_corrects_reasons = 0
        action_counts = torch.zeros(num_actions, device=device)
        action_corrects = torch.zeros(num_actions, device=device)
        '''
        for inputs, actions_gt, reasons_gt, image_names in train_loader:
            inputs = inputs.to(device)
            actions_gt = actions_gt.to(device)
            reasons_gt = reasons_gt.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            actions_pred, reasons_pred = explanation_model(inputs)
            # print("Actions pred shape:", actions_pred.shape)
            # print("Actions gt shape:", actions_gt.shape)
            # print("Reasons pred shape:", reasons_pred.shape)
            # print("Reasons gt shape:", reasons_gt.shape)
            # Calculate loss
            loss_actions = criterion(actions_pred, actions_gt)
            loss_reasons = criterion(reasons_pred, reasons_gt)
            loss = loss_actions + loss_reasons

            # Backward pass
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

            # Update running loss
        train_loss = train_running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

        # Validate
        explanation_model.eval()

        # Initialize variables for accumulating statistics
        running_val_loss = 0
        running_actions_correct = 0
        running_reasons_correct = 0
        running_actions_total = 0
        running_reasons_total = 0
        running_actions_per_class_correct = [0, 0, 0, 0]
        incorrect_image_names = []

        actions_total_f1 = []
        reasons_total_f1 = []
        actions_pred_f1 = [[], [], [], []]
        actions_gt_f1 = [[], [], [], []]

        actions_correct_images = 0
        reasons_correct_images = 0
        actions_total_images = 0
        reasons_total_images = 0

        with torch.no_grad():
            # Iterate through the validation data loader (batches of validation samples)
            for idx, (inputs, actions_gt, reasons_gt, image_names) in enumerate(val_loader):
                inputs, actions_gt, reasons_gt = inputs.to(device), actions_gt.to(device), reasons_gt.to(device)

                # Forward pass the input images through the model, getting the predicted actions and reasons (logits)
                actions_pred, reasons_pred = explanation_model(inputs)

                # Calculate the validation loss for the current batch using the criterion (BCEWithLogitsLoss)
                val_loss = criterion(actions_pred, actions_gt) + criterion(reasons_pred, reasons_gt)
                running_val_loss += val_loss.item()

                # Apply the Sigmoid activation and threshold (0.5) to the predicted action and reason logits to
                # obtain binary predictions
                actions_pred = F.sigmoid(actions_pred) > 0.5
                reasons_pred = F.sigmoid(reasons_pred) > 0.5

                # Calculate the number of correct action and reason predictions for the current batch by comparing
                # the predicted values to the ground truth values
                actions_correct = actions_pred.eq(actions_gt.bool()).sum().item()
                reasons_correct = reasons_pred.eq(reasons_gt.bool()).sum().item()
                running_actions_correct += actions_correct
                running_reasons_correct += reasons_correct
                running_actions_total += actions_gt.numel()
                running_reasons_total += reasons_gt.numel()

                # Calculate F1 score
                for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                    actions_pred_binary = actions_pred[i].cpu().numpy().astype(int)
                    actions_f1_i = f1_score(actions_gt[i].cpu().numpy(), actions_pred_binary, average='micro', zero_division=1.0)
                    actions_total_f1.append(actions_f1_i)

                    reasons_pred_binary = reasons_pred[i].cpu().numpy().astype(int)
                    reasons_f1_i = f1_score(reasons_gt[i].cpu().numpy(), reasons_pred_binary, average='micro', zero_division=1.0)
                    reasons_total_f1.append(reasons_f1_i)

                    for j in range(num_actions):
                        # action_f1_j = f1_score(actions_gt[i, j].cpu().numpy().reshape(1),actions_pred_binary[j].reshape(1), average=None)
                        actions_pred_f1[j].append(actions_pred_binary[j])
                        actions_gt_f1[j].append(actions_gt[i, j].cpu().item())
                        # actions_f1[j].append(action_f1_j)

                # Accumulate the number of correct predictions per action class
                for i in range(4):
                    running_actions_per_class_correct[i] += actions_pred[:, i].eq(actions_gt.bool()[:, i]).sum().item()

                for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                    if actions_pred[i].eq(actions_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        actions_correct_images += 1
                actions_total_images += actions_pred.size(0)
                actions_accuracy_images = actions_correct_images/actions_total_images

                for i in range(reasons_pred.size(0)):  # Iterate over images in the batch
                    if reasons_pred[i].eq(reasons_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        reasons_correct_images += 1
                reasons_total_images += reasons_pred.size(0)
                reasons_accuracy_images = reasons_correct_images/reasons_total_images

                # If the prediction is incorrect, store the image name in the incorrect_image_names list
                incorrect_idxs = np.where(np.any(actions_pred.cpu().numpy() != actions_gt.cpu().numpy(), axis=1))[0]
                incorrect_image_names.extend([image_names[i] for i in incorrect_idxs])

        # Calculate the average validation loss, action accuracy, reason accuracy, and per-class action accuracy
        val_loss_avg = running_val_loss / len(val_loader)
        action_accuracy = running_actions_correct / running_actions_total
        reason_accuracy = running_reasons_correct / running_reasons_total
        action_accuracy_per_class = [correct / running_actions_total * 4 for correct in running_actions_per_class_correct]

        # Calculate the average F1 score
        # actions_total_f1 /= len(val_loader)
        # reasons_total_f1 /= len(val_loader)
        # actions_f1 = [f1 / len(val_loader) for f1 in actions_f1]

        # Print the validation results
        print("Validate loss and accuracy proportion")
        print(f"Validation Loss: {val_loss_avg:.4f}, Action Accuracy (per action): {action_accuracy:.4f}, Reason "
              f"Accuracy (per reason): {reason_accuracy:.4f}")
        print(f"Action Accuracy per Class: {[f'{acc:.4f}' for acc in action_accuracy_per_class]}")

        print("F1 score")
        print(f'Actions Total F1: {np.mean(actions_total_f1):.4f}')
        print(f'Reasons Total F1: {np.mean(reasons_total_f1):.4f}')
        for i in range(4):
            # print(f'Action {i + 1} F1 Score: {f1_score(actions_gt_f1[i],actions_pred_f1[i], average="average"):.4f}')
            print(f'Action {i + 1} F1 Score: {np.mean(f1_score(actions_gt_f1[i], actions_pred_f1[i], average=None)):.4f}')

        print("Solid Same")
        print(f"Action accuracy (per image): {actions_accuracy_images * 100:.2f}%")
        print(f"Reason accuracy (per image): {reasons_accuracy_images * 100:.2f}%")

        # Write the incorrect image names to a text file
        with open("incorrect_image_names.txt", "w") as f:
            for img_name in incorrect_image_names:
                f.write(f"{img_name}\n")

        # Set the model back to training mode
        explanation_model.train()

        '''
            running_loss += loss.item() * inputs.size(0)

            # Update running corrects
            _, actions_pred_labels = torch.max(actions_pred, 1)
            _, reasons_pred_labels = torch.max(reasons_pred, 1)
            _, actions_gt_labels = torch.max(actions_gt, 1)
            _, reasons_gt_labels = torch.max(reasons_gt, 1)
            # print(actions_pred_labels)
            # print(reasons_pred_labels)
            # print(actions_gt_labels)
            # print(reasons_gt_labels)
            for i in range(actions_gt_labels.size(0)):
                action_counts[actions_gt_labels[i]] += 1
                if actions_pred_labels[i] == actions_gt_labels[i]:
                    action_corrects[actions_gt_labels[i]] += 1
            running_corrects_actions += torch.sum(actions_pred_labels == actions_gt_labels)
            running_corrects_reasons += torch.sum(reasons_pred_labels == reasons_gt_labels)

        # Calculate epoch loss and accuracy
        action_acc = action_corrects / action_counts
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc_actions = running_corrects_actions.double() / len(train_dataset)
        epoch_acc_reasons = running_corrects_reasons.double() / len(train_dataset)
        print(
            f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc Actions: {epoch_acc_actions:.4f} Acc Reasons: {epoch_acc_reasons:.4f}")
        for i, acc in enumerate(action_acc):
            print(f"Accuracy for action {i + 1}: {acc:.4f}")

        '''


if __name__ == '__main__':
    main()
