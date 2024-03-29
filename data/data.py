import json
import os
import networkx as nx
import numpy as np
import torch
import torchvision.transforms as transforms
from torch_geometric.data import Dataset
from torch_geometric.utils.convert import from_networkx
from PIL import Image, ImageOps
from transformers import AutoTokenizer
import pandas as pd
import config as CFG


class PrescriptionPillData(Dataset):
    def __init__(self, json_files, mode, args):
        self.args = args
        self.mode = mode
        self.json_files_test = args.json_files_test
        self.json_files = json_files # contains all json files (name only NAMENAME.json)
        self.text_sentences_tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
        self.pill_information = pd.read_csv("data/pill_information.csv")
        self.transforms = transforms.Compose([transforms.ToTensor(), ])
        self.all_pill_labels = CFG.ALL_PILL_LABELS
        self.idx_to_pill_labels = {v: k for k, v in self.all_pill_labels.items()}
        self.unseen_labels = CFG.UNSEEN_LABELS 
        self.unseen_labels_idx = [k for k, v in self.unseen_labels.items()]

    def create_graph(self, bboxes, imgw, imgh, pills_class):
        """
        bboxes: list of bounding boxes
        imgw : image width
        imgh : image height
        pills_class : list of pills class (id: 0,1,2,3)
        """
        G = nx.Graph()
        for src_idx, src_row in enumerate(bboxes):
            pills_label = torch.tensor(-1, dtype=torch.long)
            src_row['label'] = src_row['label'].lower()
            # For pill information
            text_information = ' '

            if not src_row['label']:
                src_row['label'] = 'other'
            if src_row['label'] != 'drugname':
                src_row['label'] = 'other'

            # FOR PILL - PRESCRIPTION
            if src_row['label'] == 'drugname':
                # Add pill information
                color = self.pill_information[self.pill_information['Pill'] == src_row['mapping']]['Color'].values[0]
                shape = self.pill_information[self.pill_information['Pill'] == src_row['mapping']]['Shape'].values[0]
                text_information = f"{color} {shape}"
                
                if self.all_pill_labels[src_row['mapping']] not in pills_class:
                    pills_label = torch.tensor(-1, dtype=torch.long)
                else:
                    pills_label = torch.tensor(self.all_pill_labels[src_row['mapping']], dtype=torch.long)                

            # Lấy Index của Label trong CFG.LABELS
            src_row['y'] = torch.tensor(CFG.LABELS.index(src_row['label']), dtype=torch.long)
            src_row["x_min"], src_row["y_min"], src_row["x_max"], src_row["y_max"] = src_row["box"]
            src_row['bbox'] = list(map(float, [
                                   src_row["x_min"] / imgw,
                                   src_row["y_min"] / imgh,
                                   src_row["x_max"] / imgw,
                                   src_row["y_max"] / imgh
                                   ]))

            G.add_node(
                src_idx,
                text=src_row['text'],
                text_information=text_information,
                bbox=src_row['bbox'],
                prescription_label=src_row['y'],
                pills_label_in_prescription=pills_label
            )

            src_range_x = (src_row["x_min"], src_row["x_max"])
            src_range_y = (src_row["y_min"], src_row["y_max"])

            neighbor_vert_bot = []
            neighbor_hozi_right = []

            for dest_idx, dest_row in enumerate(bboxes):
                if dest_idx == src_idx:
                    continue
                dest_row["x_min"], dest_row["y_min"], dest_row["x_max"], dest_row["y_max"] = dest_row["box"]
                dest_range_x = (dest_row["x_min"], dest_row["x_max"])
                dest_range_y = (dest_row["y_min"], dest_row["y_max"])

                # Find box in horizontal must have common x range.
                if max(src_range_x[0], dest_range_x[0]) < min(src_range_x[1], dest_range_x[1]):
                    # Find underneath box: neighbor yminx must be smaller than source ymax
                    if dest_range_y[0] >= src_range_y[1]:
                        neighbor_vert_bot.append(dest_idx)

                # Find box in horizontal must have common y range.
                if max(src_range_y[0], dest_range_y[0]) < min(src_range_y[1], dest_range_y[1]):
                    # Find right box: neighbor xmin must be smaller than source xmax
                    if dest_range_x[0] >= src_range_x[1]:
                        neighbor_hozi_right.append(dest_idx)

            neighbors = []
            if neighbor_hozi_right:
                nei = min(neighbor_hozi_right,
                          key=lambda x: bboxes[x]['x_min'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)

            if neighbor_vert_bot:
                nei = min(neighbor_vert_bot, key=lambda x: bboxes[x]['y_min'])
                neighbors.append(nei)
                G.add_edge(src_idx, nei)

        return G

    def generate_target(self, idx, pill_label_path, label2idx):
        with open(pill_label_path) as json_file:
            data = json.load(json_file)
        data_boxes = data['boxes']
        num_objs = len(data_boxes)
        
        boxes = []
        labels = []
        corresponding_labels = []
        for i in range(num_objs):
            xmin = data_boxes[i]['x']
            xmax = data_boxes[i]['x'] + data_boxes[i]['w']
            ymin = data_boxes[i]['y']
            ymax = data_boxes[i]['y'] + data_boxes[i]['h']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1) # 1 is the label for drugname
            corresponding_labels.append(label2idx[data_boxes[i]['label']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([idx])
        corresponding_labels = torch.as_tensor(corresponding_labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "corresponding_labels": corresponding_labels
        } 
        return target

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ### FOR PILL IMAGE ###
        pill_image_path = os.path.join(self.args.data_path + "pills/" + self.mode + "/imgs/", self.json_files[idx].replace(".json", ".jpg"))
        pill_label_path = os.path.join(self.args.data_path + "pills/" + self.mode + "/labels/", self.json_files[idx])
        with open(pill_image_path, 'rb') as f:
            pill_image = Image.open(f).convert("RGB")
            pill_image = ImageOps.exif_transpose(pill_image)

        if self.transforms:
            pill_image = self.transforms(pill_image)
        pill_label_generate = self.generate_target(idx, pill_label_path, self.all_pill_labels) # boxes, labels
        pill_label_only = pill_label_generate["corresponding_labels"].tolist()
        pills_images_labels_masked = torch.Tensor([1 if self.idx_to_pill_labels[i] in self.unseen_labels else 0 for i in pill_label_only])
        ### FOR PILL IMAGE ###

        ### FOR PRESCRIPTION ###
        prescription_path = os.path.join(self.args.data_path + "pres/" + self.mode + "/", self.json_files[idx])
        with open(prescription_path, "r") as json_file:
            prescription = json.load(json_file)

        G = self.create_graph(bboxes=prescription, imgw=1000,
                              imgh=1000, pills_class=pill_label_only)
        data = from_networkx(G)

        text_sentences = self.text_sentences_tokenizer(data.text, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
        text_information = self.text_sentences_tokenizer(data.text_information, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
        data.text_sentences_ids, data.text_sentences_mask = text_sentences.input_ids, text_sentences.attention_mask
        data.text_information_ids, data.text_information_mask = text_information.input_ids, text_information.attention_mask
        ### FOR TEXT PROCESS ###

        ### FOR PILL IMAGE ###
        data.pill_image = torch.unsqueeze(pill_image, 0)
        data.pill_image_label = torch.as_tensor(pill_label_only, dtype=torch.int64)
        data.pill_label_generate = [pill_label_generate]
        data.pills_images_labels_masked = torch.Tensor(pills_images_labels_masked) # for masking unseen class
        ### FOR PILL IMAGE ###

        data.text_sentences_test_ids, data.text_sentences_test_mask = text_sentences.input_ids, text_sentences.attention_mask
        data.pill_infors_test_ids, data.pill_infors_test_mask = text_information.input_ids, text_information.attention_mask
        if self.json_files_test is not None and idx < len(self.json_files_test):
            pres_test_path = os.path.join(self.args.data_path + "pres/test/", self.json_files_test[idx])
            with open(pres_test_path, "r") as json_file:
                prescription_test = json.load(json_file)
            pill_test_texts = []
            pill_test_infors = []
            for i in prescription_test:
                i['label'] = i['label'].lower()
                if i['label'] == 'drugname':
                    # get index of i['mapping'] in dictionary UNSEEN_LABELS
                    if i['mapping'] not in self.unseen_labels:
                        continue
                    unseen_index = self.unseen_labels_idx.index(i['mapping'])
                    if unseen_index < 0 * len(self.unseen_labels_idx):
                        continue

                    pill_test_texts.append(i['text'])
                    color = self.pill_information[self.pill_information['Pill'] == i['mapping']]['Color'].values[0]
                    shape = self.pill_information[self.pill_information['Pill'] == i['mapping']]['Shape'].values[0]
                    text_infor = f"{color} {shape}"
                    pill_test_infors.append(text_infor)

            if len(pill_test_texts) > 0:
                text_sentences_test = self.text_sentences_tokenizer(pill_test_texts, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
                data.text_sentences_test_ids = torch.cat((data.text_sentences_test_ids, text_sentences_test.input_ids), dim=0)
                data.text_sentences_test_mask = torch.cat((data.text_sentences_test_mask, text_sentences_test.attention_mask), dim=0)

                pill_infors_test = self.text_sentences_tokenizer(pill_test_infors, max_length=32, padding='max_length', truncation=True, return_tensors='pt')
                data.pill_infors_test_ids = torch.cat((data.pill_infors_test_ids, pill_infors_test.input_ids), dim=0)
                data.pill_infors_test_mask = torch.cat((data.pill_infors_test_mask, pill_infors_test.attention_mask), dim=0)
        return data
