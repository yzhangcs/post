# post

Several models for POS Tagging

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
$ python run.py -e --bidirectional --lstm --char --crf
```

### Arguments

```sh
$ python run.py -h
usage: run.py [-h] [--crf] [--lstm] [--char] [--bidirectional] [--embed]
              [--file FILE] [--threads THREADS]

Create Neural Network for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --crf                 use crf
  --lstm                use lstm
  --char                use char representation
  --bidirectional       use bidirectional lstm
  --embed, -e           use pretrained embedding file
  --file FILE, -f FILE  set where to store the model
  --threads THREADS, -t THREADS
                        set max num of threads
```

## Structures

```python
# BPNN
BPNN(
  (embed): Embedding(383647, 100)
  (hid): Linear(in_features=500, out_features=300, bias=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (dropout): Dropout(p=0.5)
)
# BPNN+CRF
BPNN(
  (embed): Embedding(383647, 100)
  (hid): Linear(in_features=500, out_features=300, bias=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.5)
)
# BiLSTM+CRF
LSTM(
  (embed): Embedding(383647, 100)
  (lstm): LSTM(100, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.5)
)
# BiLSTM+CHAR+CRF
LSTM(
  (embed): Embedding(383647, 100)
  (char_lstm): CharLSTM(
    (embed): Embedding(7477, 100)
    (lstm): LSTM(100, 100, batch_first=True, bidirectional=True)
  )
  (word_lstm): LSTM(300, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=32, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.6)
)
```

