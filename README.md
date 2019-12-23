# UVR-NMT

ICLR 2020: Neural Machine Translation  with universal Visual Representation

This implementation is based on [fairseq](https://github.com/pytorch/fairseq). We take en2ro NMT experiment for example.

### Requirements

- OS: macOS or Linux
- NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
- Pytorch

### Preparation

1. Download Multi30K dataset

   ```
   git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset
   ```

   Put the `task1` folder in the `data` directory, which looks like `data/task1/image_splits/train.txt`

2. Visual Features

   Pre-extracted visual features can be [downloaded from Google Drive](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf?usp=sharing) borrowed from the repo [Multi30K](https://github.com/multi30k/dataset).

   Please put the features in the main directory to ensure the `train-resnet50-avgpool.npy` file can be accessed by:  `features_resnet50/train-resnet50-avgpool.npy`

3. BPE segmentation

   Segment both the NMT dataset and Multi30K dataset with the same BPE code file (built by the NMT dataset) using the tool [subword-nmt](https://github.com/rsennrich/subword-nmt).

4. Binarize the dataset

```
TEXT=data/en-ro/
python preprocess.py --source-lang en --target-lang ro --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/en2ro --joined-dictionary --thresholdtgt 0 --thresholdsrc 0
```

### Lookup Table

After having the following two files:

1) the segmented file for multi30k training set using the same BPE code with the NMT dataset (**multi30k_train_bpe.txt**)

*Ensure the training sets of multi30k and NMT are segmented to subwords in the same manner (BPE code)*

*so the tokens in NMT dataset can be found in the lookup table built by multi30k sentence-image pairs.*

2) the source dict of en2ro dataset （**src_dict_en2ro.txt**）

*run the model using the following script, and the source dict file will be saved at the directory  `--save-dir*`

**Note this is just for getting the *dict* file. The training will be interrupted without the lookup table, which is expected. Just start the training using the script after having the lookup table.

```
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
```

put those two files (`multi30k_train_bpe.txt` and `src_dict_en2ro.txt`) in the `data` folder, then we can get the lookup table for NMT model training.

`python image_lookup.py` 

change the directory if needed:

```
parser.add_argument('--stopwords_dir', default="data/stopwords-en.txt", help='path of the stopwords-en.txt')
parser.add_argument('--src_dict_dir', default="data/src_dict_en2ro.txt", help='path of the source dict of en2ro dataset')
parser.add_argument('--src_en_dir', default="data/multi30k_train_bpe.txt", help='path of the segmented file for multi30k training set using the same bpe code with the nmt dataset (e.g., en2ro)')
parser.add_argument('--image_dir', default="data/task1/image_splits/train.txt", help='path of the image_splits of training set of multi30k')
```

The cap2image_file is the lookup table used for training NMT model.

```
parser.add_argument('--cap2image_file', default="data/cap2image_en2ro.pickle", help='output file for (topic) word to image id lookup table')
```

### Training

```
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
```

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