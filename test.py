import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, TripletLoss
import wandb
from utils.option import option
import numpy as np
import pandas as pd 
from objdetecteval.metrics import coco_metrics as cm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from utils.utils import calculate_matching_loss, build_loaders, plot_image, new_predicts_nms_thresholds, for_evaluation_matching
import config as CFG
import warnings
import os
warnings.filterwarnings("ignore")



def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model 

def train(model_localization, model_matching, train_loader, optimizer_localization, optimizer_matching, matching_criterion, graph_criterion, epoch):
    model_localization.train()
    model_matching.train()
    train_loss = []

    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        for data in train_bar:
            pre_loss = []

            data = data.to(device)
            optimizer_localization.zero_grad()
            optimizer_matching.zero_grad()

            # Localization
            pill_image = list(img.to(device) for img in data.pill_image)
            pill_label_generate = [{k: v.to(device) for k, v in t[0].items()} for t in data.pill_label_generate]
            model_localization_return = model_localization(pill_image, pill_label_generate)
            loss_localization = {'loss_classifier': model_localization_return['loss_classifier'], 'loss_box_reg': model_localization_return['loss_box_reg'], 'loss_objectness': model_localization_return['loss_objectness'], 'loss_rpn_box_reg': model_localization_return['loss_rpn_box_reg']}

            # remove loss_classifier 
            loss_localization.pop('loss_classifier')
            exit()

            bbox_img_features = model_localization_return['gt_feature']
            bbox_labels = model_localization_return['gt_label']

            # Matching
            images_projection, sentences_projection, graph_extract, sentences_information_projection = model_matching(data, bbox_img_features)

            _, max_graph_extract = torch.max(graph_extract, 1)
            graph_loss = graph_criterion(graph_extract, data.prescription_label)

            sentences_embedding_drugname = sentences_projection[max_graph_extract == 0]
            sentences_information_projection_drugname = sentences_information_projection[max_graph_extract == 0]

            sentences_labels_drugname = data.pills_label_in_prescription[max_graph_extract == 0]

            matching_loss = calculate_matching_loss(
                images_projection, sentences_embedding_drugname, sentences_labels_drugname, bbox_labels, matching_criterion, sentences_information_projection_drugname)

            losses = sum(loss for loss in loss_localization.values()) + matching_loss + graph_loss
            losses.backward()

            optimizer_localization.step()
            optimizer_matching.step()

            pre_loss.append(losses.item())
            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)


def main(args):
    ### Init wandb
    wandb.init(entity="aiotlab", project="VAIPE-PIMA-Zero-shot-Matching", group=args.run_group, name=args.run_name,  config=args, mode= "disabled")
    print(">>> Loading data...")

    # json prescriotion_path
    train_pres_list = os.listdir(args.train_data_path)
    test_pres_list = os.listdir(args.test_data_path)

    train_loader = build_loaders(train_pres_list, mode="code", batch_size=args.train_batch_size, num_workers=args.num_workers, args=args)
    val_loader = build_loaders(test_pres_list, mode="code", batch_size=args.val_batch_size, num_workers=args.num_workers, args=args)

    print(">>> Data information:")
    print("Train data size: ", len(train_loader.dataset))
    print("Val data size: ", len(val_loader.dataset))

    print(">>> Building model...")
    ### Build Localization model
    model_localization = get_model_instance_segmentation(len(CFG.ALL_PILL_LABELS)).to(device)
    params_localization = [p for p in model_localization.parameters() if p.requires_grad]

    model_matching = PrescriptionPill(args).cuda()
    matching_criterion = ContrastiveLoss()
    graph_criterion = torch.nn.NLLLoss(weight=torch.FloatTensor(CFG.labels_weight).cuda())

    optimizer_localization = torch.optim.AdamW(params_localization, lr=args.lr, weight_decay=5e-4)
    optimizer_matching = torch.optim.AdamW(model_matching.parameters(), lr=args.lr, weight_decay=5e-4)

    print(">>> Start training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model_localization, model_matching, train_loader, optimizer_localization, optimizer_matching, matching_criterion, graph_criterion, epoch)
        # accuracy = evaluate(model_localization, model_matching, val_loader, epoch)

        # wandb.log({'train_loss': train_loss, 'AP_all': accuracy['All']['AP_all'], 'AP_50': accuracy['All']['AP_all_IOU_0_50'], 'AP_75': accuracy['All']['AP_all_IOU_0_75'], 'acc_all': accuracy['All']})

    wandb.finish()

if __name__ == '__main__':
    parse_args = option()
    torch.cuda.manual_seed(parse_args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(parse_args)
