import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import ops
import pandas as pd
from data.data import PrescriptionPillData
from torch_geometric.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import math
import random

def get_model_instance_segmentation(num_classes=2):
    """
        num_classes: 2 (Background and pill)
    """
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model 

def collate_fn(batch):
    return tuple(zip(*batch))

def build_loaders(json_file, mode="code", batch_size=1, num_workers=0, shuffle=False, args=None):
    dataset = PrescriptionPillData(json_file, mode, args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn)
    return dataloader

def calculate_matching_loss(matching_criterion, images_projection, image_information_projection, sentences_projection, sentences_information_projection, text_embedding_labels, pills_images_labels, negative_ratio=None):

    loss = []
    for idx, label in enumerate(pills_images_labels):
        positive_idx = text_embedding_labels.eq(label)
        negative_idx = text_embedding_labels.ne(label)

        anchor = images_projection[idx]
        positive = sentences_projection[positive_idx]
        negative = sentences_projection[negative_idx]

        anchor_2 = image_information_projection[idx]
        positive_2 = sentences_information_projection[positive_idx]
        negative_2 = sentences_information_projection[negative_idx]

        if negative_ratio is not None:
            # get random negative samples
            negative = negative[torch.randperm(len(negative))[:math.ceil(len(negative) * negative_ratio)]]
            negative_2 = negative_2[torch.randperm(len(negative_2))[:math.ceil(len(negative_2) * negative_ratio)]]

        loss.append(matching_criterion(anchor, positive, negative))
        loss.append(matching_criterion(anchor_2, positive_2, negative_2))
    return torch.mean(torch.stack(loss))

def calculate_matching_loss_v2(image_aggregation, text_embedding_drugname, text_embedding_labels, pills_images_labels, matching_criterion, sentecnes_information_projection_drugname, negative_ratio=None):
    image_aggregation_new = image_aggregation.unsqueeze(1)
    pills_images_labels_new = pills_images_labels.unsqueeze(1)
    text_embedding_labels_new = text_embedding_labels.repeat(image_aggregation.shape[0], 1)
    text_embedding_drugname_new = text_embedding_drugname.unsqueeze(0).repeat(image_aggregation.shape[0], 1, 1)
    sentecnes_information_projection_drugname_new = sentecnes_information_projection_drugname.unsqueeze(0).repeat(image_aggregation.shape[0], 1, 1)

    positive_text = torch.zeros_like(text_embedding_drugname_new)
    negative_text = torch.zeros_like(text_embedding_drugname_new)
    positive_infor = torch.zeros_like(sentecnes_information_projection_drugname_new)
    positive_text_index = (text_embedding_labels_new == pills_images_labels_new)
    negative_text_index = (text_embedding_labels_new != pills_images_labels_new)

    positive_text[positive_text_index] = text_embedding_drugname_new[positive_text_index]
    positive_infor[positive_text_index] = sentecnes_information_projection_drugname_new[positive_text_index]
    negative_text[negative_text_index] = text_embedding_drugname_new[negative_text_index]

    return matching_criterion(image_aggregation_new, positive_text, positive_infor, negative_text)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def calculate_matching_cross_loss(images_projection, sentences_projection, sentences_information_projection, sentences_all_projection, sentences_information_all_projection, text_embedding_labels, pills_images_labels, temperature=1.0):
    images_projection_corresponding = images_projection
    sentences_projection_corresponding = torch.zeros_like(images_projection)
    sentences_information_projection_corresponding = torch.zeros_like(images_projection)

    for idx, label in enumerate(pills_images_labels):
        positive_idx = torch.where(text_embedding_labels == label)[0]
        if len(positive_idx) > 1:
            # positive_idx = positive_idx[torch.randperm(len(positive_idx))[:1]]
            positive_idx = positive_idx[0]
        sentences_projection_corresponding[idx] = sentences_projection[positive_idx]
        sentences_information_projection_corresponding[idx] = sentences_information_projection[positive_idx]

    img_zero = torch.zeros_like(images_projection[0]).to(images_projection.device)
    # for idx, label in enumerate(text_embedding_labels):
    #     if label != -1:
    #         continue
    #     if random.random() < 0.95:
    #         continue
    #     sentences_projection_corresponding = torch.cat((sentences_projection_corresponding, sentences_projection[idx].unsqueeze(0)), dim=0)
    #     images_projection_corresponding = torch.cat((images_projection_corresponding, img_zero.unsqueeze(0)), dim=0)

    # Matching image with drugname
    logits = (images_projection_corresponding @ sentences_projection_corresponding.T) / temperature
    targets = torch.arange(logits.shape[0]).to(images_projection_corresponding.device)
    matching_loss = torch.nn.functional.cross_entropy(logits, targets) + torch.nn.functional.cross_entropy(logits.T, targets)

    # Matching image with information
    logits_infor = (images_projection @ sentences_information_projection_corresponding.T) / temperature
    targets_infor = torch.arange(logits_infor.shape[0]).to(images_projection.device)
    matching_infor_loss = torch.nn.functional.cross_entropy(logits_infor, targets_infor) + torch.nn.functional.cross_entropy(logits_infor.T, targets_infor)

    # Matching all 
    logits_all = (sentences_all_projection @ sentences_information_all_projection.T) / temperature
    targets_all = torch.arange(logits_all.shape[0]).to(sentences_all_projection.device)
    matching_all_loss = torch.nn.functional.cross_entropy(logits_all, targets_all) + torch.nn.functional.cross_entropy(logits_all.T, targets_all)

    return (matching_loss + matching_infor_loss + matching_all_loss) / 6


def plot_image(img_tensor, annotation):
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.cpu().data
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.savefig("test.png")


def new_predicts_nms_thresholds(predicts, nms_threshold):
    for i in range(len(predicts)):
        nms_index = ops.nms(predicts[i]['boxes'], predicts[i]['scores'], nms_threshold)
        predicts[i]['boxes'] = predicts[i]['boxes'][nms_index]
        predicts[i]['labels'] = predicts[i]['labels'][nms_index]
        predicts[i]['scores'] = predicts[i]['scores'][nms_index]
        predicts[i]['feature'] = predicts[i]['feature'][nms_index]
    return predicts


def for_evaluation(predicts, targets, predicts_df, targets_df): 
    for predict, target in zip(predicts, targets):
        image_id = target['image_id'].item()
        
        if len(predict['boxes']) == 0:
            predicts_df = pd.concat([predicts_df, pd.DataFrame([[len(predicts_df), 0,0,0,0,-1,0,image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])
        else:
            for i in range(len(predict['boxes'])):
                id = len(predicts_df)
                xmin = predict['boxes'][i][0].item()
                ymin = predict['boxes'][i][1].item()
                xmax = predict['boxes'][i][2].item()
                ymax = predict['boxes'][i][3].item()
                label = predict['labels'][i].item()
                score = predict['scores'][i].item()

                predicts_df =pd.concat([predicts_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, score, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])

        for i in range(len(target['boxes'])):
            id = len(targets_df)
            xmin = target['boxes'][i][0].item()
            ymin = target['boxes'][i][1].item()
            xmax = target['boxes'][i][2].item()
            ymax = target['boxes'][i][3].item()
            label = target['labels'][i].item()

            targets_df = pd.concat([targets_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])])

    return predicts_df, targets_df


def for_evaluation_matching(predicts, targets, predicts_df, targets_df, mapping_predicted): 
    idx = 0
    for predict, target in zip(predicts, targets):
        image_id = target['image_id'].item()
        
        if len(predict['boxes']) == 0:
            predicts_df = pd.concat([predicts_df, pd.DataFrame([[len(predicts_df), 0,0,0,0,-1,0,image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])
        else:
            for i in range(len(predict['boxes'])):
                id = len(predicts_df)
                xmin = predict['boxes'][i][0].item()
                ymin = predict['boxes'][i][1].item()
                xmax = predict['boxes'][i][2].item()
                ymax = predict['boxes'][i][3].item()
                label = mapping_predicted[idx].item()
                idx += 1
                score = predict['scores'][i].item()

                predicts_df =pd.concat([predicts_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, score, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])

        for i in range(len(target['boxes'])):
            id = len(targets_df)
            xmin = target['boxes'][i][0].item()
            ymin = target['boxes'][i][1].item()
            xmax = target['boxes'][i][2].item()
            ymax = target['boxes'][i][3].item()
            label = target['corresponding_labels'][i].item()

            targets_df = pd.concat([targets_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])])

    return predicts_df, targets_df


def for_evaluation_matching_unseen(predicts, targets, predicts_df, targets_df, mapping_predicted, selected_label): 
    idx = 0
    for predict, target in zip(predicts, targets):
        image_id = target['image_id'].item()
        
        if len(predict['boxes']) == 0:
            predicts_df = pd.concat([predicts_df, pd.DataFrame([[len(predicts_df), 0,0,0,0,-1,0,image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])
        else:
            flag = False
            for i in range(len(predict['boxes'])):
                label = mapping_predicted[idx].item()
                idx += 1
                if label not in selected_label:
                    continue
                id = len(predicts_df)
                xmin = predict['boxes'][i][0].item()
                ymin = predict['boxes'][i][1].item()
                xmax = predict['boxes'][i][2].item()
                ymax = predict['boxes'][i][3].item()
                score = predict['scores'][i].item()

                predicts_df =pd.concat([predicts_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, score, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])
                flag = True

            if not flag:
                predicts_df = pd.concat([predicts_df, pd.DataFrame([[len(predicts_df), 0,0,0,0,-1,0,image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])])

        for i in range(len(target['boxes'])):
            label = target['corresponding_labels'][i].item()
            if label not in selected_label:
                continue
            id = len(targets_df)
            xmin = target['boxes'][i][0].item()
            ymin = target['boxes'][i][1].item()
            xmax = target['boxes'][i][2].item()
            ymax = target['boxes'][i][3].item()
            targets_df = pd.concat([targets_df, pd.DataFrame([[id, xmin, ymin, xmax, ymax, label, image_id]], columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])])

    return predicts_df, targets_df
