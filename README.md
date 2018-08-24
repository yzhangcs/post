# post

Several models for POS Tagging

Already implemented models:

* BPNN+CRF
* BiLSTM+CRF
* BiLSTM+CHAR+CRF

TODO:

* implement self-attention

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
$ python run.py -e --bi --lstm --char --crf
```

### Arguments

```sh
$ python run.py -h
usage: run.py [-h] [--crf] [--attn] [--bpnn] [--lstm] [--char] [--file FILE]
              [--threads THREADS]

Create Neural Network for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --crf                 use crf
  --attn                use attention
  --bpnn                use bpnn
  --lstm                use lstm
  --char                use char representation
  --file FILE, -f FILE  set where to store the model
  --threads THREADS, -t THREADS
                        set max num of threads
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

* https://github.com/LiyuanLucasLiu/LM-LSTM-CRF
* https://github.com/kmkurn/pytorch-crf
* https://github.com/jadore801120/attention-is-all-you-need-pytorch
* https://arxiv.org/pdf/1706.03762.pdf
* https://arxiv.org/pdf/1508.01991.pdf
* https://arxiv.org/pdf/1804.09849.pdf

