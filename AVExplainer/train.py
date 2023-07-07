import torch
from Explainer import *
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import warnings
from detectron2.checkpoint import DetectionCheckpointer
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, roc_curve
from detectron2.data import (
    MetadataCatalog
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.simplefilter(action='ignore', category=FutureWarning)
relationship_matrix = torch.tensor([
    # Actions 1-4
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1]
])


def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    model = build_model(cfg)
    # model.load_state_dict(torch.load('base_model.pth'))
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_train.json', 'r') as f:
        train_annotations = json.load(f)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_val.json', 'r') as f:
        val_annotations = json.load(f)

    train_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/train', train_annotations)
    val_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/validate', val_annotations)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    num_actions = 4
    num_reasons = 21
    explanation_model = ExplanationModel(model, num_actions, num_reasons, True, True)
    # explanation_model.load_state_dict(torch.load('edl_sep_base_model.pth'))

    '''
    pretrained_dict = torch.load('edl_sep_base_model.pth')
    # Extract the state dictionary of the selector and fc1
    selector_dict = {k: v for k, v in pretrained_dict.items() if 'selector' in k and 'selector_' not in k}
    selector_dict = {k.replace('selector.', ''): v for k, v in selector_dict.items()}
    fc1_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('fc1.')}
    fc1_dict = {k.replace('fc1.', ''): v for k, v in fc1_dict.items()}

    # Initialize the selectors with the pretrained weights
    explanation_model.selector_1.load_state_dict(selector_dict)
    explanation_model.selector_2.load_state_dict(selector_dict)
    explanation_model.selector_3.load_state_dict(selector_dict)
    explanation_model.selector_4.load_state_dict(selector_dict)
    explanation_model.fc1_1.load_state_dict(fc1_dict)
    explanation_model.fc1_2.load_state_dict(fc1_dict)
    explanation_model.fc1_3.load_state_dict(fc1_dict)
    explanation_model.fc1_4.load_state_dict(fc1_dict)

    # Get the rest of the parameters
    rest_dict = {k: v for k, v in pretrained_dict.items() if 'selector' not in k and 'fc1' not in k}
    # rest_dict = {k: v for k, v in pretrained_dict.items() if 'selector' not in k}
    # Get the current state dict of the new model
    model_dict = explanation_model.state_dict()

    # Filter out unnecessary keys from rest_dict
    rest_dict = {k: v for k, v in rest_dict.items() if k in model_dict}

    # Update the current model's state dict
    model_dict.update(rest_dict)

    # Load the updated state dict into the new model
    explanation_model.load_state_dict(model_dict)
    '''
    device = torch.device("cuda")
    explanation_model.to(device)

    # Define the loss function: normal BCE
    criterion = nn.BCEWithLogitsLoss()

    # Define the loss function: weighted BCE
    class_weights = [1, 1, 1, 1]
    w = torch.FloatTensor(class_weights).cuda()
    criterion_w = nn.BCEWithLogitsLoss(pos_weight=w).cuda()

    # Define the loss function: edl
    criterion_edl = edl_digamma_loss
    criterion_weighted_edl = edl_weighted_digamma_loss

    # Create an optimizer (base lr 0.001)
    optimizer = optim.Adam(explanation_model.parameters(), lr=0.00001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Determine the number of epochs
    num_epochs = 50
    weights_dict_1 = {}
    weights_dict_2 = {}
    weights_dict_3 = {}
    weights_dict_4 = {}
    last_training_loss = 100.0

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

        '''
        if epoch > 0:

            # Unfreeze some specific parameters
            for name, param in explanation_model.named_parameters():
                
                if name in ["selector_1.0.weight", "selector_1.0.bias", "selector_1.2.weight", "selector_1.2.bias",
                            "selector_1.4.weight", "selector_1.4.bias",
                            "selector_2.0.weight", "selector_2.0.bias", "selector_2.2.weight", "selector_2.2.bias",
                            "selector_2.4.weight", "selector_2.4.bias",
                            "selector_3.0.weight", "selector_3.0.bias", "selector_3.2.weight", "selector_3.2.bias",
                            "selector_3.4.weight", "selector_3.4.bias",
                            "selector_4.0.weight", "selector_4.0.bias", "selector_4.2.weight", "selector_4.2.bias",
                            "selector_4.4.weight", "selector_4.4.bias"
                            ]:
                    param.requires_grad = True
                
                
                if "fc2" in name:
                    param.requires_grad = False
                
            parameters_to_update = [param for param in explanation_model.parameters() if param.requires_grad]
            optimizer = optim.Adam(parameters_to_update, lr=0.00001)
        '''

        train_running_loss = 0.0
        train_uncertainties = []
        train_entropy = []

        for inputs, actions_gt, reasons_gt, image_names in train_loader:
            inputs = inputs.to(device)
            actions_gt = actions_gt.to(device)
            reasons_gt = reasons_gt.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            actions_pred, reasons_pred, actions_rel = explanation_model(inputs)

            if explanation_model.use_edl:
                # Forward pass
                actions_pred_result = torch.argmax(actions_pred, dim=2)
                # weights = torch.ones(len(actions_pred_result))
                actions_weights = torch.ones(num_actions, len(actions_pred_result))

                if epoch == 0:
                    for i in range(len(actions_pred_result)):
                        single_image_pred = actions_pred[i]
                        single_image_pred_result = actions_pred_result[i]
                        single_image_uncertainty = torch.tensor(
                            [2 / (torch.sum(relu_evidence(a) + 1)) for a in single_image_pred])
                        single_image_prob = torch.stack(
                            [(relu_evidence(a) + 1) / (torch.sum(relu_evidence(a) + 1)) for a in single_image_pred])
                        single_image_entropy = -torch.sum(single_image_prob * torch.log2(single_image_prob + 1e-9),
                                                          dim=-1)

                        for a in range(num_actions):
                            train_uncertainties.append(single_image_uncertainty[a])
                            train_entropy.append(single_image_entropy[a])

                            if single_image_pred_result[a].eq(actions_gt[i][a].bool()):
                                if a == 0:
                                    if single_image_uncertainty[a] > 0.579:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_1[image_names[i]] = actions_weights[a][i]
                                elif a == 1:
                                    if single_image_uncertainty[a] > 0.554:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_2[image_names[i]] = actions_weights[a][i]
                                elif a == 2:
                                    if single_image_uncertainty[a] > 0.53:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_3[image_names[i]] = actions_weights[a][i]
                                else:
                                    if single_image_uncertainty[a] > 0.695:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_4[image_names[i]] = actions_weights[a][i]
                            else:
                                if a == 0:
                                    if single_image_entropy[a] < 0.874:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_1[image_names[i]] = actions_weights[a][i]
                                elif a == 1:
                                    if single_image_entropy[a] < 0.851:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_2[image_names[i]] = actions_weights[a][i]
                                elif a == 2:
                                    if single_image_entropy[a] < 0.834:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_3[image_names[i]] = actions_weights[a][i]
                                else:
                                    if single_image_entropy[a] < 0.931:
                                        actions_weights[a][i] = 1.0
                                    weights_dict_4[image_names[i]] = actions_weights[a][i]
                else:
                    for i in range(num_actions):
                        for j in range(len(actions_pred_result)):
                            if i == 0:
                                if image_names[j] in weights_dict_1:
                                    actions_weights[i, j] = weights_dict_1[image_names[j]]
                            elif i == 1:
                                if image_names[j] in weights_dict_2:
                                    actions_weights[i, j] = weights_dict_2[image_names[j]]
                            elif i == 2:
                                if image_names[j] in weights_dict_3:
                                    actions_weights[i, j] = weights_dict_3[image_names[j]]
                            else:
                                if image_names[j] in weights_dict_4:
                                    actions_weights[i, j] = weights_dict_4[image_names[j]]

                    # print(actions_weights)

                split_action_pred = torch.unbind(actions_pred, dim=1)  # tuple of 4 tensors with shape 4x2
                split_reason_pred = torch.unbind(reasons_pred, dim=1)  # tuple of 21 tensors with shape 4x2
                split_action_gt = torch.unbind(actions_gt, dim=1)  # tuple of 4 tensors with shape 4,
                split_reason_gt = torch.unbind(reasons_gt, dim=1)  # tuple of 4 tensors with shape 4,

                loss_actions = 0
                loss_reasons = 0

                for i in range(num_actions):
                    action_pred_i = split_action_pred[i]
                    action_gt_i = split_action_gt[i]
                    action_gt_i = action_gt_i.to(torch.long)
                    action_gt_i = one_hot_embedding(action_gt_i)
                    loss_actions += criterion_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device)
                    #loss_actions += criterion_weighted_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device, actions_weights[i])

                for j in range(num_reasons):
                    reason_pred_i = split_reason_pred[j]
                    reason_gt_i = split_reason_gt[j]
                    reason_gt_i = reason_gt_i.to(torch.long)
                    reason_gt_i = one_hot_embedding(reason_gt_i)
                    loss_reasons += criterion_edl(reason_pred_i, reason_gt_i.float(), epoch, 2, 10, device)
                    # loss_reasons += criterion_weighted_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device, weights)

            else:
                loss_actions = criterion_w(actions_pred, actions_gt)
                loss_reasons = criterion(reasons_pred, reasons_gt)

            loss = loss_actions   + loss_reasons

            # Add regularization term to loss
            '''
            reg_lambda = 0.01  # Regularization strength, adjust as needed
            relationship_matrix_norm = torch.norm(explanation_model.relationship_matrix)
            loss += reg_lambda * relationship_matrix_norm
            '''

            # Backward pass
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

            # Update running loss
        train_loss = train_running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
        if train_loss > last_training_loss:
            print("Training loss is increasing! We should stop!")
            # break
        else:
            last_training_loss = train_loss

        '''
        if explanation_model.use_edl:
            print(f"Training Uncertainty for action 1: ", torch.stack(train_uncertainties[0::4]).mean().item())
            print(f"Training Uncertainty for action 2: ", torch.stack(train_uncertainties[1::4]).mean().item())
            print(f"Training Uncertainty for action 3: ", torch.stack(train_uncertainties[2::4]).mean().item())
            print(f"Training Uncertainty for action 4: ", torch.stack(train_uncertainties[3::4]).mean().item())

            print(f"Training Entropy for action 1: ", torch.stack(train_entropy[0::4]).mean().item())
            print(f"Training Entropy for action 2: ", torch.stack(train_entropy[1::4]).mean().item())
            print(f"Training Entropy for action 3: ", torch.stack(train_entropy[2::4]).mean().item())
            print(f"Training Entropy for action 4: ", torch.stack(train_entropy[3::4]).mean().item())
        '''
        # Validate
        explanation_model.eval()

        '''
        results_df = pd.DataFrame(
            columns=['image_name', 'action_uncertainty', 'action_probability',
                     'missed_actions', 'excessive_actions', 'correct_actions',
                     'reason_uncertainty',
                     'missed_reasons', 'excessive_reasons', 'correct_reasons'])
        '''

        with torch.no_grad():
            # Calculate F1 Separately
            actions_total_f1 = []
            reasons_total_f1 = []
            actions_pred_f1 = [[], [], [], []]
            actions_gt_f1 = [[], [], [], []]

            # Calculate solid accuracy
            actions_correct_images = 0
            reasons_correct_images = 0
            reasons_reasonable_images = 0
            actions_total_images = 0
            reasons_total_images = 0

            # Calculate Val F1
            all_actions_gt = []
            all_actions_pred = []
            all_actions_pred_auc = []
            all_actions_gt_auc = []
            all_reasons_gt = []
            all_reasons_pred = []

            # Calculate Val AUC
            val_uncertainties = []
            val_entropy = []
            val_correctness = []
            val_running_loss = 0.0

            # Iterate through the validation data loader (batches of validation samples)
            for idx, (inputs, actions_gt, reasons_gt, image_names) in enumerate(val_loader):
                inputs, actions_gt, reasons_gt = inputs.to(device), actions_gt.to(device), reasons_gt.to(device)

                # Forward pass the input images through the model, getting the predicted actions and reasons (logits)
                actions_pred, reasons_pred, _ = explanation_model(inputs)

                if explanation_model.use_edl:
                    split_action_pred = torch.unbind(actions_pred, dim=1)  # tuple of 4 tensors with shape 4x2
                    split_reason_pred = torch.unbind(reasons_pred, dim=1)  # tuple of 21 tensors with shape 4x2
                    split_action_gt = torch.unbind(actions_gt, dim=1)  # tuple of 4 tensors with shape 4,
                    split_reason_gt = torch.unbind(reasons_gt, dim=1)  # tuple of 4 tensors with shape 4,

                    loss_actions = 0
                    loss_reasons = 0

                    for i in range(num_actions):
                        action_pred_i = split_action_pred[i]
                        action_gt_i = split_action_gt[i]
                        action_gt_i = action_gt_i.to(torch.long)
                        action_gt_i = one_hot_embedding(action_gt_i)
                        loss_actions += criterion_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device)
                        # loss_actions += criterion_weighted_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device,
                        # actions_weights[i])

                    for j in range(num_reasons):
                        reason_pred_i = split_reason_pred[j]
                        reason_gt_i = split_reason_gt[j]
                        reason_gt_i = reason_gt_i.to(torch.long)
                        reason_gt_i = one_hot_embedding(reason_gt_i)
                        loss_reasons += criterion_edl(reason_pred_i, reason_gt_i.float(), epoch, 2, 10, device)
                        # loss_reasons += criterion_weighted_edl(action_pred_i, action_gt_i.float(), epoch, 2, 10, device, weights)

                    for k in range(len(actions_pred)):
                        single_action_pred = actions_pred[k]
                        single_action_pred_result = torch.argmax(actions_pred, dim=2)[k]
                        single_model_uncertainty = [(2 / torch.sum(relu_evidence(a) + 1)).item() for a in
                                                    single_action_pred]
                        single_probability = torch.stack(
                            [(relu_evidence(a) + 1) / (torch.sum(relu_evidence(a) + 1)) for a in single_action_pred])
                        single_entropy = -torch.sum(single_probability * torch.log2(single_probability + 1e-9), dim=-1)
                        for b in range(num_actions):
                            val_uncertainties.append(single_model_uncertainty[b])
                            val_entropy.append(single_entropy[b])
                            all_actions_pred_auc.append(single_action_pred[b][1])
                            all_actions_gt_auc.append(actions_gt[k][b])
                            if single_action_pred_result[b].eq(actions_gt[k][b].bool()):
                                val_correctness.append(1)
                            else:
                                val_correctness.append(0)

                        '''
                        sample_reason_pred = reasons_pred[0]
                        sample_reason_uncertainty = [(2 / torch.sum(relu_evidence(b) + 1)).item() for b in
                                                    sample_reason_pred]

                        sample_action_gt = actions_gt[0].cpu().numpy().astype(int)
                        sample_reason_gt = reasons_gt[0].cpu().numpy().astype(int)

                        sample_image_name = image_names[0]
                        '''
                    actions_pred = torch.argmax(actions_pred, dim=2)
                    reasons_pred = torch.argmax(reasons_pred, dim=2)

                    '''
                    sample_action_pred = actions_pred[0].cpu().numpy().astype(int)
                    sample_reason_pred = reasons_pred[0].cpu().numpy().astype(int)
                    missed_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if
                                      g == 1 and p == 0]
                    excessive_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if
                                         g == 0 and p == 1]
                    correct_actions = [i for i, (g, p) in enumerate(zip(sample_action_gt, sample_action_pred)) if
                                       g == p]

                    missed_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if
                                      g == 1 and p == 0]
                    excessive_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if
                                         g == 0 and p == 1]
                    correct_reasons = [i for i, (g, p) in enumerate(zip(sample_reason_gt, sample_reason_pred)) if
                                       g == p]

                    results_df = results_df.append({
                        'image_name': sample_image_name,
                        'action_uncertainty': sample_model_uncertainty,
                        'action_probability': sample_probability,
                        'missed_actions': missed_actions,
                        'excessive_actions': excessive_actions,
                        'correct_actions': correct_actions,
                        'reason_uncertainty': sample_reason_uncertainty,
                        'missed_reasons': missed_reasons,
                        'excessive_reasons': excessive_reasons,
                        'correct_reasons': correct_reasons
                    }, ignore_index=True)
                    '''
                # Apply the Sigmoid activation and threshold (0.5) to the predicted action and reason logits to
                else:
                    for k in range(len(actions_pred)):
                        single_action_pred = actions_pred[k]
                        for b in range(num_actions):
                            all_actions_pred_auc.append(single_action_pred[b])
                            all_actions_gt_auc.append(actions_gt[k][b])

                    actions_pred = F.sigmoid(actions_pred) > 0.5
                    reasons_pred = F.sigmoid(reasons_pred) > 0.5

                loss = loss_actions + loss_reasons
                val_running_loss += loss

                all_actions_pred.append(actions_pred.cpu().numpy().astype(int))
                all_actions_gt.append(actions_gt.cpu().numpy().astype(int))
                all_reasons_pred.append(reasons_pred.cpu().numpy().astype(int))
                all_reasons_gt.append(reasons_gt.cpu().numpy().astype(int))

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
                        # action_f1_j = f1_score(actions_gt[i, j].cpu().numpy().reshape(1),actions_pred_binary[
                        # j].reshape(1), average=None)
                        actions_pred_f1[j].append(actions_pred_binary[j])
                        actions_gt_f1[j].append(actions_gt[i, j].cpu().item())
                        # actions_f1[j].append(action_f1_j)

                for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                    if actions_pred[i].eq(
                            actions_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        actions_correct_images += 1
                actions_total_images += actions_pred.size(0)
                actions_accuracy_images = actions_correct_images / actions_total_images

                for i in range(reasons_pred.size(0)):
                    image_reasonable = True
                    for j in range(num_reasons):
                        if reasons_pred[i, j] == 1:  # If the reason is present
                            if not any([actions_pred[i, k] == 1 and relationship_matrix[k, j] == 1 for k in
                                        range(num_actions)]):  # If there is no corresponding action
                                image_reasonable = False
                                break
                    if image_reasonable:
                        reasons_reasonable_images += 1

                for i in range(reasons_pred.size(0)):  # Iterate over images in the batch
                    if reasons_pred[i].eq(
                            reasons_gt[i].bool()).all():  # Check if all actions are correct for the current image
                        reasons_correct_images += 1
                reasons_total_images += reasons_pred.size(0)
                reasons_accuracy_images = reasons_correct_images / reasons_total_images
                reasons_reasonable_rate = reasons_reasonable_images / reasons_total_images

        # Calculate loss
        val_loss = val_running_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        print("Exactly Same Answer Rate: ")
        print(f"Action accuracy (per image): {actions_accuracy_images * 100:.2f}%")
        print(f"Reason accuracy (per image): {reasons_accuracy_images * 100:.2f}%")
        print(f"Reason reasonable rate (per image): {reasons_reasonable_rate * 100:.2f}%")

        # Concatenate all the batch-wise true labels and predictions
        all_actions_pred = np.concatenate(all_actions_pred, axis=0)
        all_actions_gt = np.concatenate(all_actions_gt, axis=0)
        all_reasons_pred = np.concatenate(all_reasons_pred, axis=0)
        all_reasons_gt = np.concatenate(all_reasons_gt, axis=0)

        # Now calculate the F1 score over the entire dataset
        f1_actions_score_overall = f1_score(all_actions_gt, all_actions_pred, average='samples')
        f1_reasons_score_overall = f1_score(all_reasons_gt, all_reasons_pred, average='samples')
        f1_actions_score_macro = f1_score(all_actions_gt, all_actions_pred, average='macro')
        f1_reasons_score_macro = f1_score(all_reasons_gt, all_reasons_pred, average='macro')
        f1_actions_score_micro = f1_score(all_actions_gt, all_actions_pred, average='micro')
        f1_reasons_score_micro = f1_score(all_reasons_gt, all_reasons_pred, average='micro')
        f1_actions_score = f1_score(all_actions_gt, all_actions_pred, average=None)
        # f1_reasons_score = f1_score(all_reasons_gt, all_reasons_pred, average=None)
        print("F1 score")
        print(f'Actions Total F1: {f1_actions_score_overall:.4f}')
        print(f'Reasons Total F1: {f1_reasons_score_overall:.4f}')
        print(f'Actions Macro F1: {f1_actions_score_macro:.4f}')
        print(f'Reasons Macro F1: {f1_reasons_score_macro:.4f}')
        print(f'Actions micro F1: {f1_actions_score_micro:.4f}')
        print(f'Reasons micro F1: {f1_reasons_score_micro:.4f}')
        print(f'Actions F1: ', f1_actions_score)

        # Calculate the AUCs
        aucs = []
        all_actions_pred_auc = [tensor.cpu().numpy() for tensor in all_actions_pred_auc]
        all_actions_gt_auc = [tensor.cpu().numpy() for tensor in all_actions_gt_auc]
        val_entropy = [tensor.cpu().numpy() for tensor in val_entropy]
        for a in range(num_actions):
            # auc = roc_auc_score(val_correctness[a::num_actions], val_uncertainties[a::num_actions])
            auc = roc_auc_score(all_actions_gt_auc[a::num_actions], all_actions_pred_auc[a::num_actions])
            aucs.append(auc)
        auc_toal = roc_auc_score(all_actions_gt_auc, all_actions_pred_auc)
        print("AUC Score")
        print(f'AUC for each action: ', aucs)
        print(f'AUC for total: ', auc_toal)

        if aucs[0] >= 0.809 and aucs[1] >= 0.816 and aucs[2] >= 0.73 and aucs[3] >= 0.718:
            model_name = "test_ori_epoch_" + str(epoch) + ".pth"
            torch.save(explanation_model.state_dict(), model_name)

        # Determine the threshold that maximizes the AUC
        if explanation_model.use_edl:
            thresholds_uncertainty = []
            thresholds_entropy = []
            for a in range(num_actions):
                fpr, tpr, thrs = roc_curve(val_correctness[a::num_actions], val_uncertainties[a::num_actions])
                gmeans = np.sqrt(tpr * (1 - fpr))  # geometric mean of true positive rate and true negative rate
                idx = np.argmax(gmeans)  # Index of maximum gmean
                thresholds_uncertainty.append(thrs[idx])

            for a in range(num_actions):
                fpr, tpr, thrs = roc_curve(val_correctness[a::num_actions], val_entropy[a::num_actions])
                gmeans = np.sqrt(tpr * (1 - fpr))  # geometric mean of true positive rate and true negative rate
                idx = np.argmax(gmeans)  # Index of maximum gmean
                thresholds_entropy.append(thrs[idx])
            print("Threshold")
            print(f'Threshold for model uncertainty: ', thresholds_uncertainty)
            print(f'Threshold entropy: ', thresholds_entropy)

        # results_df.to_csv('uncertainty_results.csv', index=False)

        # Set the model back to training mode
        explanation_model.train()
        scheduler.step()

    torch.save(explanation_model.state_dict(), 'test_edl_sep_model.pth')


if __name__ == '__main__':
    main()
