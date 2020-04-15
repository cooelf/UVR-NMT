TEXT=data/en-ro/
python preprocess.py --source-lang en --target-lang ro --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/en2ro --joined-dictionary --thresholdtgt 0 --thresholdsrc 0
DATA_DIR=data-bin/en2ro/
python train.py ${DATA_DIR} --task translation \
      --arch transformer_wmt_en_de --share-all-embeddings --dropout 0.1 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
      --lr 0.0007 --min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
      --max-tokens 4096\
      --update-freq 2 --no-progress-bar --log-format json --log-interval 50 \
      --save-interval-updates 1000 --keep-interval-updates 500 --max-update 100000 --source-lang en --target-lang ro \
      --save-dir checkpoints/base-en2ro \
      --save_src_dict data/src_dict_en2ro.txt \
      --cap2image_file data/cap2image_en2ro.pickle \
      --image_embedding_file features_resnet50/train-resnet50-avgpool.npy \
      --encoder-type TransformerAvgEncoder \
      --L2norm true --image_emb_fix --total_num_img 5 --per_num_img 1 --find-unused-parameters --merge_option att-gate --gate_type neural-gate