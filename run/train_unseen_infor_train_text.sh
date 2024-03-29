CUDA_VISIBLE_DEVICES=1 python train.py --run-name="Mobilenet-V3 + Infor (Train Text)" --run-group="Unseen"  --matching-criterion="ContrastiveLoss" --text-model-name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --text-embedding=384 --train-batch-size=16 --val-batch-size=1 --num-workers=16 --data-path="/workdir/data/VAIPE-PP-Unseen-01/" --unseen-training=True --text-trainable=True
