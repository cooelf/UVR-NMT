# UVR-NMT

ICLR 2020: Neural Machine Translation  with universal Visual Representation

This implementation is based on [fairseq](https://github.com/pytorch/fairseq). We take en2de NMT experiment for example.

*working in progress

### Requirements

- OS: macOS or Linux
- NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
- Pytorch

### Preparation

1. Download Multi30K dataset

   ```
   git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset
   ```

2. Visual Features

   Pre-extracted visual features can be [downloaded from Google Drive](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf?usp=sharing) borrowed from the repo [Multi30K](https://github.com/multi30k/dataset).

   The features are used in image embedding layer for indexing. 

3. Data-Preprocessing

   Segment both the NMT dataset and Multi30K dataset with the same BPE code file (built by the NMT dataset) using the tool [subword-nmt](https://github.com/rsennrich/subword-nmt).

   Run `prepare-wmt-en2de.sh --icml17` (for WMT'14 En-De), which is a modified version of [prepare-wmt14en2de.sh](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh)

### Lookup Table

Before generating the lookup table, the following two files should be prepared:

1) the segmented file for multi30k training set using the same BPE code with the NMT dataset (**wmt14_en_de/bpe.multi30k.en**)

*Ensure the training sets of multi30k and NMT are segmented to subwords in the same manner (BPE code)*

*so the tokens in NMT dataset can be found in the lookup table built by multi30k sentence-image pairs.*

2) the source dict of WMT dataset （**data/src_dict_wmt_en2de.txt**）

*run the model using the following script, and the source dict file will be saved at the directory  `--save-dir*`

**Note this is just for getting the *dict* file. The training will be interrupted without the lookup table, which is expected. Just start the training using the script after having the lookup table.

```
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
```

Then we can get the lookup table for NMT model training.

​	`bash sh_en2de_map.sh`

```
python image_lookup.py \
--src_dict_dir data/src_dict_wmt_en2de.txt \
--src_en_dir wmt14_en_de/bpe.multi30k.en \
--image_dir multi30k-dataset/data/task1/image_splits/train.txt \
--cap2image_file data/cap2image_en2de.pickle
```

change the directory if needed:

```
parser.add_argument('--stopwords_dir', default="data/stopwords-en.txt", help='path of the stopwords-en.txt')
parser.add_argument('--src_dict_dir', default="data/src_dict_wmt_en2de.txt", help='path of the source dict of WMT dataset')
parser.add_argument('--src_en_dir', default="wmt14_en_de/bpe.multi30k.en", help='path of the segmented file for multi30k training set using the same bpe code with the nmt dataset (e.g., en2de)')
parser.add_argument('--image_dir', default="multi30k-dataset/data/task1/image_splits/train.txt", help='path of the image_splits of training set of multi30k')
```

The cap2image_file is the lookup table used for training NMT model.

```
parser.add_argument('--cap2image_file', default="data/cap2image_en2de.pickle", help='output file for (topic) word to image id lookup table')
```

### Training

```
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
```

### Inference

```
DATA_DIR=data-bin/wmt14_en2de
MODEL_DIR=checkpoints/base-wmt-en2de/checkpoint_best.pt
TEXT=wmt14_en_de

CUDA_VISIBLE_DEVICES=0 python interactive.py ${DATA_DIR} --input ${TEXT}/test.en \
 --path ${MODEL_DIR} --beam 5 --remove-bpe --lenpen 0.6  \
 --source-lang en --target-lang de --batch-size 64 --buffer-size 1000 > ./result/wmt14_ende_test.pred
```

### Application

A trained WMT'14 En2De model can be downloaded [from here](https://drive.google.com/open?id=1cRwFiT0nWJq2gWecMof8I5Mna3BFnlVx).

### Reference

Please kindly cite this paper in your publications if it helps your research:

```
@inproceedings{zhang2020neural,
title={Neural Machine Translation with Universal Visual Representation},
author={Zhuosheng Zhang and Kehai Chen and Rui Wang and Masao Utiyama and Eiichiro Sumita and Zuchao Li and Hai Zhao},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=Byl8hhNYPS}
}
```
