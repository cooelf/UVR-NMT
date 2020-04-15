TEXT=wmt14_en_de
python preprocess.py --source-lang en --target-lang de --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/wmt14_en2de --joined-dictionary --thresholdtgt 0 --thresholdsrc 0 --workers 20
DATA_DIR=data-bin/wmt14_en2de/
python train.py ${DATA_DIR} --task translation \
      --arch transformer_wmt_en_de --share-all-embeddings --dropout 0.15 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
      --lr 0.0007 --min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
      --max-tokens 4096\
      --update-freq 1 --no-progress-bar --log-format json --log-interval 100 \
      --save-interval-updates 1000 --keep-interval-updates 1000 --max-update 300000 --source-lang en --target-lang de \
      --save-dir checkpoints/base-wmt-en2de \
      --save_src_dict data/src_dict_wmt_en2de.txt \
      --cap2image_file data/cap2image_en2de.pickle \
      --image_embedding_file features_resnet50/train-resnet50-avgpool.npy \
      --encoder-type TransformerAvgEncoder --L2norm true --image_emb_fix --total_num_img 5 --per_num_img 1 --find-unused-parameters --merge_option att-gate --gate_type neural-gate