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
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_val.json', 'r') as f:
        val_annotations = json.load(f)

    with open('C:/Users/lsion/Desktop/lastframe/gt_4a_21r_test.json', 'r') as f:
        test_annotations = json.load(f)

    val_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/validate', val_annotations)
    test_dataset = CustomDataset('C:/Users/lsion/Desktop/lastframe/test', test_annotations)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    num_actions = 4
    num_reasons = 21
    explanation_model = ExplanationModel(model, num_actions, num_reasons, True, True)
    explanation_model.load_state_dict(torch.load('edl_sep_base_model.pth'))
    device = torch.device("cuda")
    explanation_model.to(device)

    explanation_model.eval()
    results_df = pd.DataFrame(
        columns=['image_name', 'action_pred', 'action_gt'])
    with torch.no_grad():
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
        all_reasons_pred_auc = []
        all_reasons_gt_auc = []
        all_reasons_gt = []
        all_reasons_pred = []
        all_reasonable_rates_while_actions_correct = []

        # Calculate Val AUC
        val_uncertainties = []
        val_entropy = []
        val_correctness = []

        # Iterate through the validation data loader (batches of validation samples)
        for idx, (inputs, actions_gt, reasons_gt, image_names) in enumerate(test_loader):
            inputs, actions_gt, reasons_gt = inputs.to(device), actions_gt.to(device), reasons_gt.to(device)

            # Forward pass the input images through the model, getting the predicted actions and reasons (logits)
            actions_pred, reasons_pred, _ = explanation_model(inputs)

            if explanation_model.use_edl:

                for k in range(len(actions_pred)):
                    single_action_pred = actions_pred[k]
                    single_action_pred_result = torch.argmax(actions_pred, dim=2)[k]
                    single_model_uncertainty = [(2 / torch.sum(relu_evidence(a) + 1)).item() for a in
                                                single_action_pred]
                    single_probability = torch.stack(
                        [(relu_evidence(a) + 1) / (torch.sum(relu_evidence(a) + 1)) for a in single_action_pred])
                    single_entropy = -torch.sum(single_probability * torch.log2(single_probability + 1e-9), dim=-1)
                    image_name = image_names[k]
                    single_action_pred_cpu = single_action_pred.cpu().numpy()
                    action_gt_cpu = actions_gt[k].cpu().numpy()
                    results_df = results_df.append({
                        'image_name': image_name,
                        'action_pred': single_action_pred_cpu,
                        'action_gt': action_gt_cpu
                    }, ignore_index=True)
                    for b in range(num_actions):
                        val_uncertainties.append(single_model_uncertainty[b])
                        val_entropy.append(single_entropy[b])
                        all_actions_pred_auc.append(single_action_pred[b][1])
                        all_actions_gt_auc.append(actions_gt[k][b])
                        if single_action_pred_result[b].eq(actions_gt[k][b].bool()):
                            val_correctness.append(1)
                        else:
                            val_correctness.append(0)

                for i in range(len(reasons_pred)):
                    single_reason_pred = reasons_pred[i]
                    for j in range(num_reasons):
                        all_reasons_pred_auc.append(single_reason_pred[j][1])
                        all_reasons_gt_auc.append(reasons_gt[i][j])

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
                    image_name = image_names[k]
                    single_action_pred_cpu = single_action_pred.cpu().numpy()
                    action_gt_cpu = actions_gt[k].cpu().numpy()
                    results_df = results_df.append({
                        'image_name': image_name,
                        'action_pred': single_action_pred_cpu,
                        'action_gt': action_gt_cpu
                    }, ignore_index=True)
                    for b in range(num_actions):
                        all_actions_pred_auc.append(single_action_pred[b])
                        all_actions_gt_auc.append(actions_gt[k][b])

                actions_pred = F.sigmoid(actions_pred) > 0.5
                reasons_pred = F.sigmoid(reasons_pred) > 0.5

            all_actions_pred.append(actions_pred.cpu().numpy().astype(int))
            all_actions_gt.append(actions_gt.cpu().numpy().astype(int))
            all_reasons_pred.append(reasons_pred.cpu().numpy().astype(int))
            all_reasons_gt.append(reasons_gt.cpu().numpy().astype(int))

            for i in range(actions_pred.size(0)):  # Iterate over images in the batch
                present_reasons = 0.0
                reasonble_reasons = 0.0
                if actions_pred[i].eq(
                        actions_gt[i].bool()).all():  # Check if all actions are correct for the current image
                    actions_correct_images += 1
                    for j in range(num_reasons):
                        if reasons_pred[i, j] == 1:
                            present_reasons += 1
                            if any([actions_pred[i, k] == 1 and relationship_matrix[k, j] == 1 for k in
                                range(num_actions)]):
                                reasonble_reasons += 1
                if present_reasons != 0.0:
                    all_reasonable_rates_while_actions_correct.append(reasonble_reasons/present_reasons)
                else:
                    all_reasonable_rates_while_actions_correct.append(0.0)

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

    print('Exactly Same Answer Rate: ')
    print(f"Action accuracy (per image): {actions_accuracy_images * 100:.2f}%")
    print(f"Reason accuracy (per image): {reasons_accuracy_images * 100:.2f}%")
    print(f"Reason reasonable rate (per image): {reasons_reasonable_rate * 100:.2f}%")
    print(f"Reasons reasinable rate while actions correct: ", sum(all_reasonable_rates_while_actions_correct) / len(all_reasonable_rates_while_actions_correct))

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
    aucs_reasons = []
    all_actions_pred_auc = [tensor.cpu().numpy() for tensor in all_actions_pred_auc]
    all_actions_gt_auc = [tensor.cpu().numpy() for tensor in all_actions_gt_auc]
    all_reasons_pred_auc = [tensor.cpu().numpy() for tensor in all_reasons_pred_auc]
    all_reasons_gt_auc = [tensor.cpu().numpy() for tensor in all_reasons_gt_auc]
    val_entropy = [tensor.cpu().numpy() for tensor in val_entropy]
    for a in range(num_actions):
        # auc = roc_auc_score(val_correctness[a::num_actions], val_uncertainties[a::num_actions])
        auc = roc_auc_score(all_actions_gt_auc[a::num_actions], all_actions_pred_auc[a::num_actions])
        aucs.append(auc)
    auc_toal = roc_auc_score(all_actions_gt_auc, all_actions_pred_auc)
    for b in range(num_reasons):
        auc = roc_auc_score(all_reasons_gt_auc[b::num_reasons], all_reasons_pred_auc[b::num_reasons])
        aucs_reasons.append(auc)
    auc_reasons_total = roc_auc_score(all_reasons_gt_auc, all_reasons_pred_auc)
    print("AUC Score")
    print(f'AUC for each action: ', aucs)
    print(f'AUC for total actions: ', auc_toal)
    print(f'AUC for each reason: ', aucs_reasons)
    print(f'AUC for total reasons: ', auc_reasons_total)

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
    results_df.to_csv("output_v4.csv", index=False)


if __name__ == '__main__':
    main()