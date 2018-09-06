# post

Several models for POS Tagging

Already implemented models:

* BPNN+CRF
* BiLSTM+CRF
* BiLSTM+CHAR+CRF

TODO:

* implement the encoder described by [this paper](https://arxiv.org/pdf/1804.09849.pdf)
* optimize the Viterbi algorithm

## Requirements

```txt
python == 3.6.5
pytorch == 0.4.1
```

## Usage

### Commands

```sh
$ git clone https://github.com/zysite/post.git
$ cd post
# eg: BiLSTM+CHAR+CRF
$ python run.py --model=lstm_char --crf
```

### Arguments

```sh
$ python run.py -h
usage: run.py [-h] [--model {bpnn,lstm,lstm_char}] [--crf] [--prob PROB]
              [--batch_size BATCH_SIZE] [--epochs EPOCHS]
              [--interval INTERVAL] [--eta ETA] [--threads THREADS]
              [--seed SEED] [--file FILE]

Create several models for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --model {bpnn,lstm,lstm_char}, -m {bpnn,lstm,lstm_char}
                        choose the model for POS Tagging
  --crf                 use crf
  --prob PROB           set the prob of dropout
  --batch_size BATCH_SIZE
                        set the size of batch
  --epochs EPOCHS       set the max num of epochs
  --interval INTERVAL   set the max interval to stop
  --eta ETA             set the learning rate of training
  --threads THREADS, -t THREADS
                        set the max num of threads
  --seed SEED, -s SEED  set the seed for generating random numbers
  --file FILE, -f FILE  set where to store the model
```

## Structures

```python
# BPNN
BPNN(
  (embed): Embedding(54303, 100)
  (hid): Sequential(
    (0): Linear(in_features=500, out_features=300, bias=True)
    (1): ReLU()
  )
  (out): Linear(in_features=300, out_features=32, bias=True)
  (drop): Dropout(p=0.5)
  (lossfn): CrossEntropyLoss()
)
# BPNN+CRF
BPNN(
  (embed): Embedding(54303, 100)
  (hid): Sequential(
    (0): Linear(in_features=500, out_features=300, bias=True)
    (1): ReLU()
  )
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
  (lossfn): CrossEntropyLoss()
)
# BiLSTM+CRF
LSTM(
  (embed): Embedding(54303, 100)
  (lstm): LSTM(100, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
  (lossfn): CrossEntropyLoss()
)
# BiLSTM+CHAR+CRF
LSTM_CHAR(
  (embed): Embedding(54303, 100)
  (clstm): CharLSTM(
    (embed): Embedding(7477, 100)
    (lstm): LSTM(100, 100, batch_first=True, bidirectional=True)
  )
  (wlstm): LSTM(300, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
  (lossfn): CrossEntropyLoss()
)
```

## References

* [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)
* [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
* [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
* [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/pdf/1804.09849.pdf)

