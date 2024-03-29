import torch
from torch import nn
from torchvision import models
import timm
from torch_geometric.nn import SAGEConv
from transformers import AutoModel
import torch.nn.functional as F


class SBERTxSAGE(torch.nn.Module):
    def __init__(self, dropout_rate, input_dim, output_dim):
        super().__init__()
        self.hidden_size = input_dim
        self.conv1 = SAGEConv(self.hidden_size, 512)
        self.conv2 = SAGEConv(512,  output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data, pooled_output):
        edge_index, edge_weight = data.edge_index, data.edge_attr
        x = pooled_output
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                      p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class sentencesTransformer(torch.nn.Module):
    def __init__(self, model_name, trainable):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = trainable

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text_sentences_ids, text_sentences_mask):
        model_output = self.model(text_sentences_ids, text_sentences_mask)
        return self.mean_pooling(model_output, text_sentences_mask)


class ImageEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, trainable=False, image_num_classes=None):
        super().__init__()
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = trainable
            if image_num_classes is not None:
                self.model.fc = nn.Linear(
                    self.model.fc.in_features, image_num_classes)
            else:
                self.model.fc = nn.Identity()
        elif model_name == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
            for param in self.model.parameters():
                param.requires_grad = trainable
            self.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ImageEncoderTimm(nn.Module):
    def __init__(self, image_model_name="resnet50", image_pretrained=False, image_trainable=False):
        super().__init__()
        self.model = timm.create_model(
            image_model_name, image_pretrained, num_classes=0, global_pool="avg")
        for param in self.model.parameters():
            param.requires_grad = image_trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
