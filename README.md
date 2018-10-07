# post

Several models for POS Tagging

Already implemented models:

* BPNN+CRF
* BiLSTM+CRF
* CHAR+BiLSTM+CRF

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
usage: run.py [-h] [--model {bpnn_crf,lstm_crf,char_lstm_crf}] [--drop DROP]
              [--batch_size BATCH_SIZE] [--epochs EPOCHS]
              [--interval INTERVAL] [--eta ETA] [--threads THREADS]
              [--seed SEED] [--file FILE]

Create several models for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --model {bpnn_crf,lstm_crf,char_lstm_crf}, -m {bpnn_crf,lstm_crf,char_lstm_crf}
                        choose the model for POS Tagging
  --drop DROP           set the prob of dropout
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
# BPNN+CRF
BPNN_CRF(
  (embed): Embedding(54304, 100)
  (hid): Sequential(
    (0): Linear(in_features=500, out_features=150, bias=True)
    (1): ReLU()
  )
  (out): Linear(in_features=150, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
# BiLSTM+CRF
LSTM_CRF(
  (embed): Embedding(54304, 100)
  (lstm_crf): LSTM(100, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
# CHAR+BiLSTM+CRF
CHAR_LSTM_CRF(
  (embed): Embedding(54304, 100)
  (clstm): CharLSTM(
    (embed): Embedding(7478, 100)
    (lstm_crf): LSTM(100, 100, batch_first=True, bidirectional=True)
  )
  (wlstm): LSTM(300, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
```

## References

* [tagger](https://github.com/glample/tagger)
* [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF)
* [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
* [Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)
* [Empower Sequence Labeling with Task-Aware Neural Language Model](https://arxiv.org/pdf/1709.04109.pdf)