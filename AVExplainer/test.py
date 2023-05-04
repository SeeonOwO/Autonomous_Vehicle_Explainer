from Explainer import *
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import warnings
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model = build_model(cfg)
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    '''
    # Your existing code for setting up the data loaders and iterating through the train_loader
    transform = transforms.Compose([
        transforms.Resize((736, 1280)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    '''

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_train.json', 'r') as f:
        train_annotations = json.load(f)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_val.json', 'r') as f:
        val_annotations = json.load(f)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_test.json', 'r') as f:
        test_annotations = json.load(f)

    train_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/train', train_annotations)
    val_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/validate', val_annotations)
    test_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/test', test_annotations)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    num_actions = 4
    num_reasons = 21
    explanation_model = ExplanationModel(model, num_actions, num_reasons, True)
    device = torch.device("cuda")
    explanation_model.to(device)

    # Define the loss function: normal BCE
    criterion = nn.BCEWithLogitsLoss()

    # Define the loss function: weighted BCE
    class_weights = [1, 1, 2, 2]
    w = torch.FloatTensor(class_weights).cuda()
    criterion_w = nn.BCEWithLogitsLoss(pos_weight=w).cuda()

    # Define the loss function: edl
    # criterion_edl = edl_mse_loss
    criterion_edl = edl_digamma_loss

    # Create an optimizer
    optimizer = optim.Adam(explanation_model.parameters(), lr=0.001)

    # Determine the number of epochs
    num_epochs = 10
    for epoch in range(num_epochs):

        # Freeze pretrained parts
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

        for inputs, actions_gt, reasons_gt, image_names in train_loader:
            inputs = inputs.to(device)
            actions_gt = actions_gt.to(device)
            reasons_gt = reasons_gt.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            actions_pred, reasons_pred = explanation_model(inputs)

            if explanation_model.use_edl:
                split_action_pred = torch.unbind(actions_pred, dim=1) # tuple of 4 tensors with shape 4x2
                split_reason_pred = torch.unbind(reasons_pred, dim=1) # tuple of 21 tensors with shape 4x2
                split_action_gt = torch.unbind(actions_gt, dim=1) # tuple of 4 tensors with shape 4,
                split_reason_gt = torch.unbind(reasons_gt, dim=1) # tuple of 4 tensors with shape 4,

                loss_actions = 0
                loss_reasons = 0

                for i in range(num_actions):
                    action_pred_i = split_action_pred[i]
                    action_gt_i = split_action_gt[i]
                    action_gt_i = action_gt_i.to(torch.long)
                    action_gt_i = one_hot_embedding(action_gt_i)
                    loss_actions += criterion_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device)

                for j in range(num_reasons):
                    reason_pred_i = split_reason_pred[j]
                    reason_gt_i = split_reason_gt[j]
                    reason_gt_i = reason_gt_i.to(torch.long)
                    reason_gt_i = one_hot_embedding(reason_gt_i)
                    loss_actions += criterion_edl(reason_pred_i, reason_gt_i.float(), epoch, 2, 10, device)

            else:
                loss_actions = criterion_w(actions_pred, actions_gt)
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

        # running_val_loss = 0

        '''
        # Initialize variables for accumulating statistics
        running_actions_correct = 0
        running_reasons_correct = 0
        running_actions_total = 0
        running_reasons_total = 0
        running_actions_per_class_correct = [0, 0, 0, 0]
        '''

        actions_total_f1 = []
        reasons_total_f1 = []
        actions_pred_f1 = [[], [], [], []]
        actions_gt_f1 = [[], [], [], []]

        actions_correct_images = 0
        reasons_correct_images = 0
        actions_total_images = 0
        reasons_total_images = 0

        results_df = pd.DataFrame(
            columns=['image_name', 'action_uncertainty', 'action_probability',
                     'missed_actions', 'excessive_actions', 'correct_actions',
                     'missed_reasons', 'excessive_reasons', 'correct_reasons'])

        all_model_uncertainties = []
        all_data_uncertainties = []

        with torch.no_grad():
            # Iterate through the validation data loader (batches of validation samples)
            for idx, (inputs, actions_gt, reasons_gt, image_names) in enumerate(val_loader):
                inputs, actions_gt, reasons_gt = inputs.to(device), actions_gt.to(device), reasons_gt.to(device)

                # Forward pass the input images through the model, getting the predicted actions and reasons (logits)
                actions_pred, reasons_pred = explanation_model(inputs)

                '''
                # Calculate the validation loss for the current batch using the criterion (BCEWithLogitsLoss)
                val_loss = criterion(actions_pred, actions_gt) + criterion(reasons_pred, reasons_gt)
                running_val_loss += val_loss.item()
                '''

                if explanation_model.use_edl:
                    sample_action_pred = actions_pred[0]
                    sample_model_uncertainty = [(2 / torch.sum(relu_evidence(a) + 1)).item() for a in sample_action_pred]
                    sample_probability = [((relu_evidence(a) + 1) / torch.sum(relu_evidence(a) + 1)).tolist() for a in
                                          sample_action_pred]
                    sample_action_gt = actions_gt[0].cpu().numpy().astype(int)
                    sample_reason_gt = reasons_gt[0].cpu().numpy().astype(int)
                    sample_image_name = image_names[0]


                    actions_pred = torch.argmax(actions_pred, dim=2)
                    reasons_pred = torch.argmax(reasons_pred, dim=2)

                    sample_action_pred = actions_pred[0].cpu().numpy().astype(int)
                    sample_reason_pred = reasons_pred[0].cpu().numpy().astype(int)
                    missed_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if g == 1 and p == 0]
                    excessive_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if g == 0 and p == 1]
                    correct_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if g == p]

                    missed_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if g == 1 and p == 0]
                    excessive_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if g == 0 and p == 1]
                    correct_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if g == p]

                    results_df = results_df.append({
                        'image_name': sample_image_name,
                        'action_uncertainty': sample_model_uncertainty,
                        'action_probability': sample_probability,
                        'missed_actions': missed_actions,
                        'excessive_actions': excessive_actions,
                        'correct_actions': correct_actions,
                        'missed_reasons': missed_reasons,
                        'excessive_reasons': excessive_reasons,
                        'correct_reasons': correct_reasons
                    }, ignore_index=True)

                # Test new loss
                # Apply the Sigmoid activation and threshold (0.5) to the predicted action and reason logits to
                # obtain binary predictions
                else:
                    actions_pred = F.sigmoid(actions_pred) > 0.5
                    reasons_pred = F.sigmoid(reasons_pred) > 0.5

                '''
                # Calculate the number of correct action and reason predictions for the current batch by comparing
                # the predicted values to the ground truth values
                actions_correct = actions_pred.eq(actions_gt.bool()).sum().item()
                reasons_correct = reasons_pred.eq(reasons_gt.bool()).sum().item()
                running_actions_correct += actions_correct
                running_reasons_correct += reasons_correct
                running_actions_total += actions_gt.numel()
                running_reasons_total += reasons_gt.numel()
                '''

                # Calculate F1 score
                for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                    actions_pred_binary = actions_pred[i].cpu().numpy().astype(int)
                    actions_f1_i = f1_score(actions_gt[i].cpu().numpy(), actions_pred_binary, average='micro',
                                            zero_division=1.0)
                    actions_total_f1.append(actions_f1_i)

                    reasons_pred_binary = reasons_pred[i].cpu().numpy().astype(int)
                    reasons_f1_i = f1_score(reasons_gt[i].cpu().numpy(), reasons_pred_binary, average='micro',
                                            zero_division=1.0)
                    reasons_total_f1.append(reasons_f1_i)

                    for j in range(num_actions):
                        # action_f1_j = f1_score(actions_gt[i, j].cpu().numpy().reshape(1),actions_pred_binary[j].reshape(1), average=None)
                        actions_pred_f1[j].append(actions_pred_binary[j])
                        actions_gt_f1[j].append(actions_gt[i, j].cpu().item())
                        # actions_f1[j].append(action_f1_j)

                '''
                # Accumulate the number of correct predictions per action class
                for i in range(4):
                    running_actions_per_class_correct[i] += actions_pred[:, i].eq(actions_gt.bool()[:, i]).sum().item()
                '''
                for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                    if actions_pred[i].eq(
                            actions_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        actions_correct_images += 1
                actions_total_images += actions_pred.size(0)
                actions_accuracy_images = actions_correct_images / actions_total_images

                for i in range(reasons_pred.size(0)):  # Iterate over images in the batch
                    if reasons_pred[i].eq(
                            reasons_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        reasons_correct_images += 1
                reasons_total_images += reasons_pred.size(0)
                reasons_accuracy_images = reasons_correct_images / reasons_total_images

                '''
                for img_name, mu_batch, du_batch in zip(image_names, model_uncertainty, data_uncertainty):
                    # Find missed and excessive reasons
                    idx = image_names.index(img_name)
                    gt = reasons_gt.cpu().numpy()[idx]
                    pred = reasons_pred.cpu().numpy()[idx]

                    missed = [i for i, (g, p) in enumerate(zip(gt, pred)) if g == 1 and p == 0]
                    excessive = [i for i, (g, p) in enumerate(zip(gt, pred)) if g == 0 and p == 1]
                    correct = [i for i, (g, p) in enumerate(zip(gt, pred)) if g == p]

                    # Add the information to the DataFrame
                    results_df = results_df.append({
                        'image_name': img_name,
                        'action_model_uncertainty': mu_batch,
                        'action_data_uncertainty': du_batch,
                        'missed_reasons': missed,
                        'excessive_reasons': excessive,
                        'correct_reasons': correct
                    }, ignore_index=True)
                '''
        '''
        # Calculate the average validation loss, action accuracy, reason accuracy, and per-class action accuracy
        val_loss_avg = running_val_loss / len(val_loader)
        action_accuracy = running_actions_correct / running_actions_total
        reason_accuracy = running_reasons_correct / running_reasons_total
        action_accuracy_per_class = [correct / running_actions_total * 4 for correct in
                                     running_actions_per_class_correct]

        
        # Print the validation results
        print("Validate loss and accuracy proportion")
        print(f"Validation Loss: {val_loss_avg:.4f}, Action Accuracy (per action): {action_accuracy:.4f}, Reason "
              f"Accuracy (per reason): {reason_accuracy:.4f}")
        print(f"Action Accuracy per Class: {[f'{acc:.4f}' for acc in action_accuracy_per_class]}")
        '''

        print("F1 score")
        print(f'Actions Total F1: {np.mean(actions_total_f1):.4f}')
        print(f'Reasons Total F1: {np.mean(reasons_total_f1):.4f}')
        for i in range(4):
            print(
                f'Action {i + 1} F1 Score: {np.mean(f1_score(actions_gt_f1[i], actions_pred_f1[i], average=None)):.4f}')

        print("Exactly Same Answer Rate: ")
        print(f"Action accuracy (per image): {actions_accuracy_images * 100:.2f}%")
        print(f"Reason accuracy (per image): {reasons_accuracy_images * 100:.2f}%")

        results_df.to_csv('uncertainty_results.csv', index=False)

        '''
        all_model_uncertainties = np.array(all_model_uncertainties)
        all_data_uncertainties = np.array(all_data_uncertainties)
        high_uncertainty_threshold_model = np.percentile(all_model_uncertainties, 75)
        high_uncertainty_threshold_data = np.percentile(all_data_uncertainties, 75)
        print("model threshold 75%: ", high_uncertainty_threshold_model)
        print("data threshold 75%: ", high_uncertainty_threshold_data)
        '''

        # Set the model back to training mode
        explanation_model.train()


if __name__ == '__main__':
    main()
