import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, TripletLoss
import wandb
from utils.option import option
import pandas as pd 
from objdetecteval.metrics import coco_metrics as cm
from utils.utils import calculate_matching_loss, calculate_matching_cross_loss, build_loaders, plot_image, new_predicts_nms_thresholds, for_evaluation_matching, for_evaluation_matching_unseen, get_model_instance_segmentation
import config as CFG
import warnings
import os
warnings.filterwarnings("ignore")


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

            bbox_img_features = model_localization_return['gt_feature']
            # Matching
            images_projection, sentences_projection, sentences_information_projection, sentences_all_projection, sentences_information_all_projection, graph_extract = model_matching(data, bbox_img_features)

            ## Graph module
            graph_loss = graph_criterion(graph_extract, data.prescription_label)

            graph_predict = torch.nn.functional.softmax(graph_extract, dim=-1)
            graph_predict = graph_predict[:, 0].unsqueeze(1)
            # torch where graph_predict > 0.8 then 1 else 0
            # graph_predict = torch.where(graph_predict > 0.9, torch.ones_like(graph_predict), torch.zeros_like(graph_predict))
            sentences_projection = graph_predict * sentences_projection

            ## Matching module
            matching_loss = calculate_matching_cross_loss(images_projection, sentences_projection, sentences_information_projection, sentences_all_projection, sentences_information_all_projection, data.pills_label_in_prescription, data.pill_image_label)

            losses = sum(loss for loss in loss_localization.values()) + matching_loss + graph_loss
            losses.backward()

            optimizer_localization.step()
            optimizer_matching.step()

            pre_loss.append(losses.item())
            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)

def evaluate(model_localization, model_matching, val_loader): 
    model_localization.eval()
    model_matching.eval()

    predicts_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])
    targets_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.cuda()
            # Localization
            pill_image = list(img.to(device) for img in data.pill_image)
            pill_label_generate = [{k: v.to(device) for k, v in t[0].items()} for t in data.pill_label_generate]
            model_localization_return = model_localization(pill_image)
            # model_localization_return = new_predicts_nms_thresholds(model_localization_return, 0.5)

            # print(model_localization_return)
            bbox_img_features = [i['feature'] for i in model_localization_return]
            bbox_img_features = torch.cat(bbox_img_features, dim=0)

            # Matching
            images_projection, sentences_projection, sentences_information_projection, graph_extract = model_matching(data, bbox_img_features)

            similarity_text_matching = torch.nn.functional.softmax(images_projection @ sentences_projection.t(
            ), dim=1) + torch.nn.functional.softmax(images_projection @ sentences_information_projection.t(), dim=1)

            _, predicted = torch.max(similarity_text_matching, 1)
            mapping_predicted = data.pills_label_in_prescription[predicted]

            predicts_df, targets_df = for_evaluation_matching(model_localization_return, pill_label_generate, predicts_df, targets_df, mapping_predicted)

    # save predicts_df, targets_df
    accuracy = cm.get_coco_from_dfs(predicts_df, targets_df, False)
    return accuracy


def evaluate_unseen(model_localization, model_matching, val_loader): 
    model_localization.eval()
    model_matching.eval()

    predicts_seen_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])
    predicts_unseen_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])
    targets_seen_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])
    targets_unseen_df = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_name'])

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.cuda()
            # Localization
            pill_image = list(img.to(device) for img in data.pill_image)
            pill_label_generate = [{k: v.to(device) for k, v in t[0].items()} for t in data.pill_label_generate]
            model_localization_return = model_localization(pill_image)
            # model_localization_return = new_predicts_nms_thresholds(model_localization_return, 0.5)

            # print(model_localization_return)
            bbox_img_features = [i['feature'] for i in model_localization_return]
            bbox_img_features = torch.cat(bbox_img_features, dim=0)

            # Matching
            images_projection, sentences_projection, _, _, _, graph_extract = model_matching(data, bbox_img_features)

            graph_predict = torch.nn.functional.softmax(graph_extract, dim=-1)
            graph_predict = graph_predict[:, 0].unsqueeze(1)
            # graph_predict = torch.where(graph_predict > 0.9, torch.ones_like(graph_predict), torch.zeros_like(graph_predict))
            sentences_projection = graph_predict * sentences_projection

            similarity_text_matching = torch.nn.functional.softmax(images_projection @ sentences_projection.t(), dim=1)

            _, predicted = torch.max(similarity_text_matching, 1)
            mapping_predicted = data.pills_label_in_prescription[predicted]

            if sum(data.pills_images_labels_masked == 0) > 0:
                seen_label = data.pill_image_label[data.pills_images_labels_masked == 0]
                predicts_seen_df, targets_seen_df = for_evaluation_matching_unseen(model_localization_return, pill_label_generate, predicts_seen_df, targets_seen_df, mapping_predicted, seen_label)
            
            if sum(data.pills_images_labels_masked == 1) > 0:
                unseen_label = data.pill_image_label[data.pills_images_labels_masked == 1]
                predicts_unseen_df, targets_unseen_df = for_evaluation_matching_unseen(model_localization_return, pill_label_generate, predicts_unseen_df, targets_unseen_df, mapping_predicted, unseen_label)

    accuracy_seen = cm.get_coco_from_dfs(predicts_seen_df, targets_seen_df, False)
    accuracy_unseen = cm.get_coco_from_dfs(predicts_unseen_df, targets_unseen_df, False)
    return accuracy_seen, accuracy_unseen

def main(args):
    ### Init wandb
    wandb.init(entity="aiotlab", project="VAIPE-PIMA-Zero-shot-Matching", group=args.run_group, name=args.run_name, config=args) # mode= "disabled",
    print(">>> Loading data...")

    # json prescriotion_path
    train_pres_list = os.listdir(args.data_path + 'pres/train/')
    test_pres_list = os.listdir(args.data_path + 'pres/test/')
    args.json_files_test = test_pres_list

    train_loader = build_loaders(train_pres_list, mode="train", batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, args=args)
    val_loader = build_loaders(test_pres_list, mode="test", batch_size=args.val_batch_size, num_workers=args.num_workers, args=args)

    print(">>> Data information:")
    print("Train data size: ", len(train_loader.dataset))
    print("Val data size: ", len(val_loader.dataset))

    print(">>> Building model...")
    ### Build Localization model
    model_localization = get_model_instance_segmentation().to(device)

    model_matching = PrescriptionPill(args).cuda()
    matching_criterion = ContrastiveLoss()
    graph_criterion = torch.nn.NLLLoss(weight=torch.FloatTensor(CFG.labels_weight).cuda())

    optimizer_localization = torch.optim.AdamW(model_localization.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_matching = torch.optim.AdamW(model_matching.parameters(), lr=args.lr, weight_decay=5e-4)

    print(">>> Start training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model_localization, model_matching, train_loader, optimizer_localization, optimizer_matching, matching_criterion, graph_criterion, epoch)

        if args.unseen_training:
            accuracy_seen, accuracy_unseen = evaluate_unseen(model_localization, model_matching, val_loader)
            wandb.log({'train_loss': train_loss, 'acc_seen': accuracy_seen['All'], 'acc_unseen': accuracy_unseen['All'],
                        'mean': (accuracy_seen['All']['AP_all'] + accuracy_unseen['All']['AP_all']) / 2})
        else:
            accuracy = evaluate(model_localization, model_matching, val_loader)
            wandb.log({'train_loss': train_loss, 'acc_all': accuracy['All']})

    wandb.finish()

if __name__ == '__main__':
    parse_args = option()
    torch.cuda.manual_seed(parse_args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(parse_args)
