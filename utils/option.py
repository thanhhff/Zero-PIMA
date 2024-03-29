import argparse
import torch


def option():
    parser = argparse.ArgumentParser()

    # Wandb
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--run-group', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    # Data Init
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--train-batch-size', type=int, default=4)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)

    # Module Image
    parser.add_argument('--image-embedding', type=int, default=1024)

    # Module Text
    parser.add_argument('--text-model-name', type=str, default="sentence-transformers/paraphrase-mpnet-base-v2")
    parser.add_argument('--text-embedding', type=int, default=768)
    parser.add_argument('--text-encoder-model', type=str,
                        default="bert-base-cased")
    parser.add_argument('--text-pretrained', type=bool, default=False)
    parser.add_argument('--text-trainable', type=bool, default=False)

    # Module Graph
    parser.add_argument('--graph-embedding', type=int, default=256)

    # for projection head; used for both image and graph encoder
    parser.add_argument('--num-projection-layers', type=int, default=1)
    parser.add_argument('--projection-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)

    # matching
    parser.add_argument('--matching-criterion', type=str, default="ContrastiveLoss")
    parser.add_argument('--negative-ratio', type=float, default=None)

    # Model Save
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--save-folder', type=str, default="logs/weights/")

    # Unseen training
    parser.add_argument('--unseen-training', type=bool, default=False)

    # With nms
    # parser.add_argument('--use-nms', type=bool, default=False)
    
    return parser.parse_args()
