# PI2NLI
This repository is for the paper Paraphrase Identification via Textual Inference. In *Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (\*SEM 2024)*, Mexico City, Mexico. Association for Computational Linguistics.

[[Paper](https://github.com/ShiningLab/PI2NLI/blob/main/assets/paper.pdf)] [[Poster](https://github.com/ShiningLab/PI2NLI/blob/main/assets/poster.pdf)] [[Slides](https://github.com/ShiningLab/PI2NLI/blob/main/assets/slides.pdf)]

## Dependencies
Ensure you have the following dependencies installed:
+ python >= 3.11.9
+ torch >= 2.3.1
+ lightning >= 2.3.0
+ transformers >= 4.41.2
+ wandb >= 0.17.2
+ rich >= 13.7.1

## Directory
```
PI2NLI
├── README.md
├── assets
├── config.py
├── main.py
├── requirements.txt
├── res
│   ├── ckpts
│   ├── data
│   │   ├── README.md
│   │   ├── all.pkl
│   │   ├── mrpc.pkl
│   │   ├── parade.pkl
│   │   ├── paws_qqp.pkl
│   │   ├── paws_wiki.pkl
│   │   ├── pit.pkl
│   │   ├── qqp.pkl
│   │   └── twitterurl.pkl
│   ├── lm
│   │   ├── README.md
│   │   ├── roberta-large
│   │   ├── roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
│   │   ├── xlnet-large-cased
│   │   └── xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli
│   ├── log
│   └── results
└── src
    ├── datamodule.py
    ├── dataset.py
    ├── eval.py
    ├── helper.py
    ├── models.py
    └── trainer.py
```

## Setups
It is recommended to use a virtual environment to manage dependencies. Follow the steps below to set up the environment and install the required packages:
```sh
$ cd PI2NLI
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Run
Before training, review and modify the training configurations in config.py as needed:
```
$ vim config.py
$ python main.py
```

## Outputs
If all goes well, you should see progress similar to the output below:
```
$ python main.py
Some weights of the model checkpoint at ./res/lm/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another ta
sk or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect t
o be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2024-06-18 19:28:47 | Logger initialized.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
Seed set to 0
2024-06-18 19:28:47 | *Configurations:*
2024-06-18 19:28:47 |   seed: 0
2024-06-18 19:28:47 |   method: mut_pi2nli
2024-06-18 19:28:47 |   data: mrpc
2024-06-18 19:28:47 |   model: roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli
2024-06-18 19:28:47 |   init_classifier: True
2024-06-18 19:28:47 |   test0shot: False
2024-06-18 19:28:47 |   max_length: 156
2024-06-18 19:28:47 |   load_ckpt: False
2024-06-18 19:28:47 |   train_batch_size: 32
2024-06-18 19:28:47 |   eval_batch_size: 64
2024-06-18 19:28:47 |   max_epochs: -1
2024-06-18 19:28:47 |   num_workers: 8
2024-06-18 19:28:47 |   learning_rate: 1e-05
2024-06-18 19:28:47 |   weight_decay: 0.001
2024-06-18 19:28:47 |   adam_epsilon: 1e-08
2024-06-18 19:28:47 |   key_metric: val_f1
2024-06-18 19:28:47 |   patience: 6
...
Trainable params: 355 M
Non-trainable params: 0
Total params: 355 M
Total estimated model params size (MB): 1.4 K
Epoch 0/-2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 191/191 0:01:23 • 0:00:00 2.26it/s v_num: 3szp train_step_loss: 0.400 Metric val_f1 improved. New best score: 0.931
Epoch 0, global step 191: 'val_f1' reached 0.93056 (best 0.93056), saving model to 'PI2NLI/res/ckpts/mut_pi
2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=0-step=191-val_f1=0.9306.ckpt' as top 1
Epoch 1/-2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 191/191 0:01:24 • 0:00:00 2.24it/s v_num: 3szp train_step_loss: 0.266 Epoch 1
, global step 382: 'val_f1' was not in top 1
Epoch 2/-2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 191/191 0:01:25 • 0:00:00 2.23it/s v_num: 3szp train_step_loss: 0.393 Metric
val_f1 improved by 0.000 >= min_delta = 0.0. New best score: 0.931
Epoch 2, global step 573: 'val_f1' reached 0.93073 (best 0.93073), saving model to 'PI2NLI/res/ckpts/mut_pi
2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=2-step=573-val_f1=0.9307.ckpt' as top 1
...
Epoch 11, global step 2292: 'val_f1' was not in top 1
Epoch 11/-2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 191/191 0:01:24 • 0:00:00 2.24it/s v_num: 3szp train_step_loss: 0.012
2024-06-18 19:48:54 | Start testing...
Restoring states from the checkpoint path at PI2NLI/res/ckpts/mut_pi2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=5-step=1146-val_f1=0.9312.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at PI2NLI/res/ckpts/mut_pi2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=5-step=1146-val_f1=0.9312.ckpt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8968116044998169     │
│          test_f1          │    0.9230769276618958     │
│        test_pos_f1        │    0.9643340706825256     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 27/27 0:00:13 • 0:00:00 1.98it/s
Restoring states from the checkpoint path at PI2NLI/res/ckpts/mut_pi2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=5-step=1146-val_f1=0.9312.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from the checkpoint at PI2NLI/res/ckpts/mut_pi2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0/epoch=5-step=1146-val_f1=0.9312.ckpt
Predicting ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 27/27 0:00:13 • 0:00:00 1.98it/s
2024-06-18 19:49:29 | |pred_acc:89.6812|pred_pos_acc:93.1125|pred_neg_acc:82.8720|pred_f1:92.3077|pred_pos_f1:96.4334|
2024-06-18 19:49:29 | Results saved as ./res/results/mut_pi2nli/mrpc/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/0.pkl.
2024-06-18 19:49:29 | Done.
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
TODO
```