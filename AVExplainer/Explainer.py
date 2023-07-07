import torch
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
    def __init__(self, detectron_model, num_actions, num_reasons, use_edl=False, select=False):
        super(ExplanationModel, self).__init__()
        self.detectron_model = detectron_model
        self.num_actions = num_actions
        self.num_reasons = num_reasons
        self.use_edl = use_edl
        self.select_num = 10
        self.select = select

        # Use the Fast R-CNN model's backbone for global feature extraction
        self.backbone = detectron_model.backbone

        self.backbone_before_fpn = self.backbone.bottom_up

        self.rpn = detectron_model.proposal_generator

        # Use the Fast R-CNN model's ROI head for local feature extraction
        self.roi_heads = detectron_model.roi_heads

        self.no_object_detected_tensor = torch.zeros((10, 256, 7, 7)).to(torch.device("cuda"))

        # Use the average pool to downgrade dimension
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Use the selector to calculate the scores for each combined features

        self.selector = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Seperate selector
        '''
        self.selector_1 = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.selector_2 = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.selector_3 = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.selector_4 = nn.Sequential(
            nn.Conv2d(2304, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        '''

        # Prevent overfit
        self.dropout = nn.Dropout(p=0.2)

        # Direct connected
        self.action_fc = nn.Linear(2304, num_actions)
        self.reason_fc = nn.Linear(2304, num_reasons)
        self.action_fc_edl = nn.Linear(2304, num_actions*2)
        self.reason_fc_edl = nn.Linear(2304, num_reasons*2)


        # Indirect connected
        self.fc1 = nn.Linear(2048 + 256, 256)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256 * self.select_num, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, num_actions)
        self.fcr = nn.Linear(256 * self.select_num, num_reasons)
        self.fc3_edl = nn.Linear(64, num_actions*2)
        self.fcr_edl = nn.Linear(256 * self.select_num, num_reasons*2)
        '''

        #Seperate network for actions
        self.fc1_1 = nn.Linear(2048 + 256, 256)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.fc2_1 = nn.Linear(256 * self.select_num, 64)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.fc3_edl_1 = nn.Linear(64, 2)

        self.fc1_2 = nn.Linear(2048 + 256, 256)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.fc2_2 = nn.Linear(256 * self.select_num, 64)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.fc3_edl_2 = nn.Linear(64, 2)

        self.fc1_3 = nn.Linear(2048 + 256, 256)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.fc2_3 = nn.Linear(256 * self.select_num, 64)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.fc3_edl_3 = nn.Linear(64, 2)

        self.fc1_4 = nn.Linear(2048 + 256, 256)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.fc2_4 = nn.Linear(256 * self.select_num, 64)
        self.relu2_4 = nn.ReLU(inplace=True)
        self.fc3_edl_4 = nn.Linear(64, 2)

        #Seperate network for reasons
        self.fcr_edl_1 = nn.Linear(256 * self.select_num, 6)
        self.fcr_edl_2 = nn.Linear(256 * self.select_num, 12)
        self.fcr_edl_3 = nn.Linear(256 * self.select_num, 6)
        self.fcr_edl_4 = nn.Linear(256 * self.select_num, 6)
        self.fcr_edl_5 = nn.Linear(256 * self.select_num, 6)
        self.fcr_edl_6 = nn.Linear(256 * self.select_num, 6)
        '''
        '''
        # Direct connected
        self.fc = nn.Linear(2304 * 7 * 7, 2048)
        '''
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
            if object_features_i.size(0) == 0:
                combined_features_i = torch.cat(
                    [self.no_object_detected_tensor, global_features_i.expand(self.no_object_detected_tensor.size(0), -1, -1, -1)], dim=1)
            else:
                combined_features_i = torch.cat(
                    [object_features_i, global_features_i.expand(object_features_i.size(0), -1, -1, -1)], dim=1)
            combined_features_per_image.append(combined_features_i)


        '''
        # Attention weights and selected features per action
        selected_features_per_action = []
        # Loop over each selector
        for selector in [self.selector_1, self.selector_2, self.selector_3, self.selector_4]:

            attention_weights_per_image = []
            for combined_features_i in combined_features_per_image:
                selector_output = selector(combined_features_i)
                pooled_output = torch.mean(selector_output, dim=(2, 3))
                scores = F.softmax(pooled_output.squeeze(1), dim=0)
                attention_weights_per_image.append(scores)

            selected_features_per_image = []
            for i, combined_features_i in enumerate(combined_features_per_image):
                attention_scores_i = attention_weights_per_image[i]
                # Sort attention scores and features in descending order
                sorted_scores, indices = torch.sort(attention_scores_i, descending=True)
                sorted_features = combined_features_i[indices]
                # If less than k features, repeat them
                if sorted_features.shape[0] < self.select_num:
                    sorted_features = sorted_features.repeat(int(self.select_num / sorted_features.shape[0]) + 1, 1, 1,
                                                             1)
                sorted_features = sorted_features[:self.select_num]
                # Normalize by number of selected features
                sorted_features /= self.select_num
                selected_features_per_image.append(sorted_features)

            # Add the computed weights and features to their respective lists
            selected_features_per_action.append(selected_features_per_image)
        '''
        attention_weights_per_image = []
        for combined_features_i in combined_features_per_image:
            selector_output = self.selector(combined_features_i)
            pooled_output = torch.mean(selector_output, dim=(2, 3))
            scores = F.softmax(pooled_output.squeeze(1), dim=0)
            attention_weights_per_image.append(scores)

        # Compute the weighted average of object features based on attention weights
        weighted_features_per_image = []
        for i, combined_features_i in enumerate(combined_features_per_image):
            attention_weights_i = attention_weights_per_image[i].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: (num_objects, 1, 1, 1)
            weighted_features_i = attention_weights_i * combined_features_i  # Element-wise multiplication
            weighted_average_i = torch.sum(weighted_features_i, dim=0, keepdim=True)  # Shape: (1, 2304, 7, 7)
            weighted_features_per_image.append(weighted_average_i)

        selected_features_per_image = []
        for i, combined_features_i in enumerate(combined_features_per_image):
            attention_scores_i = attention_weights_per_image[i]
            # Sort attention scores and features in descending order
            sorted_scores, indices = torch.sort(attention_scores_i, descending=True)
            sorted_features = combined_features_i[indices]
            # If less than k features, repeat them
            if sorted_features.shape[0] < self.select_num:
                sorted_features = sorted_features.repeat(int(self.select_num / sorted_features.shape[0]) + 1, 1, 1, 1)
            sorted_features = sorted_features[:self.select_num]
            # Normalize by number of selected features
            sorted_features /= self.select_num
            selected_features_per_image.append(sorted_features)


        if self.select:

            final_feature = torch.stack(selected_features_per_image)
            final_feature = self.global_avg_pool(final_feature)
            final_feature = torch.squeeze(final_feature)
            final_feature = final_feature.view(-1, 2304)
            final_feature = self.dropout(self.relu1(self.fc1(final_feature)))
            final_feature = final_feature.view(-1, 10, 256)
            final_feature = final_feature.view(x.shape[0], -1)  # reshape to (batch_size, 2560)
            #final_feature_1 = final_feature
            #final_feature_2 = final_feature
            #final_feature_3 = final_feature
            #final_feature_4 = final_feature
            '''
            final_feature_1 = torch.stack(selected_features_per_action[0])
            final_feature_1 = self.global_avg_pool(final_feature_1)
            final_feature_1 = torch.squeeze(final_feature_1)
            final_feature_1 = final_feature_1.view(-1, 2304)
            final_feature_1 = self.dropout(self.relu1(self.fc1(final_feature_1)))
            final_feature_1 = final_feature_1.view(-1, 10, 256)
            final_feature_1 = final_feature_1.view(x.shape[0], -1)  # reshape to (batch_size, 2560)

            final_feature_2 = torch.stack(selected_features_per_action[1])
            final_feature_2 = self.global_avg_pool(final_feature_2)
            final_feature_2 = torch.squeeze(final_feature_2)
            final_feature_2 = final_feature_2.view(-1, 2304)
            final_feature_2 = self.dropout(self.relu1(self.fc1(final_feature_2)))
            final_feature_2 = final_feature_2.view(-1, 10, 256)
            final_feature_2 = final_feature_2.view(x.shape[0], -1)  # reshape to (batch_size, 2560)

            final_feature_3 = torch.stack(selected_features_per_action[2])
            final_feature_3 = self.global_avg_pool(final_feature_3)
            final_feature_3 = torch.squeeze(final_feature_3)
            final_feature_3 = final_feature_3.view(-1, 2304)
            final_feature_3 = self.dropout(self.relu1(self.fc1(final_feature_3)))
            final_feature_3 = final_feature_3.view(-1, 10, 256)
            final_feature_3 = final_feature_3.view(x.shape[0], -1)  # reshape to (batch_size, 2560)

            final_feature_4 = torch.stack(selected_features_per_action[3])
            final_feature_4 = self.global_avg_pool(final_feature_4)
            final_feature_4 = torch.squeeze(final_feature_4)
            final_feature_4 = final_feature_4.view(-1, 2304)
            final_feature_4 = self.dropout(self.relu1(self.fc1(final_feature_4)))
            final_feature_4 = final_feature_4.view(-1, 10, 256)
            final_feature_4 = final_feature_4.view(x.shape[0], -1)  # reshape to (batch_size, 2560)
            '''
            if self.use_edl:
                reasons = self.fcr_edl(final_feature).view(x.shape[0], self.num_reasons, 2)
                final_feature = self.dropout(self.relu2(self.fc2(final_feature)))
                actions = self.dropout(self.fc3_edl(final_featur                                                                                                                                                                        e)).view(x.shape[0], self.num_actions, 2)

                '''
                reason_1 = self.fcr_edl_1(final_feature_1).view(x.shape[0], 3, 2)
                reason_2 = self.fcr_edl_2(final_feature_2).view(x.shape[0], 6, 2)
                reason_3 = self.fcr_edl_3(final_feature_3).view(x.shape[0], 3, 2)
                reason_4 = self.fcr_edl_4(final_feature_4).view(x.shape[0], 3, 2)
                reason_5 = self.fcr_edl_5(final_feature_3).view(x.shape[0], 3, 2)
                reason_6 = self.fcr_edl_6(final_feature_4).view(x.shape[0], 3, 2)

                final_feature_1 = self.dropout(self.relu2_1(self.fc2_1(final_feature_1)))
                action_1 = self.dropout(self.fc3_edl_1(final_feature_1)).view(x.shape[0], 1, 2)

                final_feature_2 = self.dropout(self.relu2_2(self.fc2_2(final_feature_2)))
                action_2 = self.dropout(self.fc3_edl_2(final_feature_2)).view(x.shape[0], 1, 2)

                final_feature_3 = self.dropout(self.relu2_3(self.fc2_3(final_feature_3)))
                action_3 = self.dropout(self.fc3_edl_3(final_feature_3)).view(x.shape[0], 1, 2)

                final_feature_4 = self.dropout(self.relu2_4(self.fc2_4(final_feature_4)))
                action_4 = self.dropout(self.fc3_edl_4(final_feature_4)).view(x.shape[0], 1, 2)

                actions = torch.cat((action_1, action_2, action_3, action_4), dim=1)
                reasons = torch.cat((reason_1, reason_2, reason_3, reason_4, reason_5, reason_6), dim=1)
                '''
                return actions, reasons, None

            else:
                reasons = self.fcr(final_feature)
                final_feature = self.dropout(self.relu2(self.fc2(final_feature)))
                actions = self.dropout(self.fc3(final_feature))
                return actions, reasons, None


        else:
            final_feature = torch.cat(selected_features_per_image, dim=0)  # Shape: (batch_size, 2304, 7, 7)
            final_feature = self.global_avg_pool(final_feature)

            flattened_feature_maps = final_feature.view(final_feature.shape[0], -1)
            flattened_feature_maps = self.dropout(flattened_feature_maps)

            if self.use_edl:
                '''
                actions = self.action_fc_edl(flattened_feature_maps).view(x.shape[0], self.num_actions, 2)
                reasons = self.reason_fc_edl(flattened_feature_maps).view(x.shape[0], self.num_reasons, 2)

                # The relationship_matrix needs to be reshaped and repeated to match the batch size
                relationship_matrix_reshaped = self.relationship_matrix.unsqueeze(0)  # add an extra dimension for batch size, shape: [1, 4, 21]
                relationship_matrix_reshaped = relationship_matrix_reshaped.repeat(x.shape[0], 1, 1)  # shape: [4, 4, 21]

                # We need to perform the multiplication for each sample in the batch separately
                actions_rel = []
                for i in range(x.shape[0]):
                    action_rel_single = torch.matmul(relationship_matrix_reshaped[i], reasons[i])  # shape: [4, 2]
                    actions_rel.append(action_rel_single)

                # Concatenate all results
                actions_rel = torch.stack(actions_rel)  # shape: [4, 4, 2]

                # Combine with initial actions prediction
                alpha_sigmoid = torch.sigmoid(self.alpha)
                actions = actions * alpha_sigmoid + (1 - alpha_sigmoid) * actions_rel
                return actions, reasons, actions_rel
                '''

            else:
                actions = self.action_fc(flattened_feature_maps)
                reasons = self.reason_fc(flattened_feature_maps)
                return actions, reasons, None
