from torch import nn
import torch.nn.functional as F
from models import ProjectionHead, sentencesTransformer, SBERTxSAGE


class PrescriptionPill(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_encoder = SBERTxSAGE(
            input_dim=args.text_embedding, output_dim=args.graph_embedding, dropout_rate=args.dropout)

        self.sentences_encoder = sentencesTransformer(
            model_name=args.text_model_name, trainable=args.text_trainable)

        self.image_projection = ProjectionHead(
            embedding_dim=args.image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.sentences_projection = ProjectionHead(
            embedding_dim=args.text_embedding, projection_dim=args.projection_dim, dropout=args.dropout)

        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine=False),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 2),
            nn.GELU()
        )

    def forward_graph(self, data, sentences_feature):
        # Getting graph embedding
        graph_features = self.graph_encoder(data, sentences_feature)
        graph_extract = self.post_process_layers(graph_features)
        graph_extract = F.log_softmax(graph_extract, dim=-1)
        return graph_extract


    def forward(self, data, bbox_img_features):
        # TEXT
        sentences_feature = self.sentences_encoder(data.text_sentences_ids, data.text_sentences_mask)
        sentences_information = self.sentences_encoder(data.text_information_ids, data.text_information_mask)

        sentences_feature_all = self.sentences_encoder(data.text_sentences_test_ids, data.text_sentences_test_mask)
        sentences_information_all = self.sentences_encoder(data.pill_infors_test_ids, data.pill_infors_test_mask)

        images_projection = self.image_projection(bbox_img_features)
        sentences_projection = self.sentences_projection(sentences_feature)
        sentences_information_projection = self.sentences_projection(sentences_information)

        sentences_all_projection = self.sentences_projection(sentences_feature_all)
        sentences_information_all_projection = self.sentences_projection(sentences_information_all)

        # GRAPH
        graph_extract = self.forward_graph(data, sentences_feature)

        return images_projection, sentences_projection, sentences_information_projection, sentences_all_projection, sentences_information_all_projection, graph_extract
