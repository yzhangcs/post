# POST

## Requirements

```txt
python == 3.6.5
pytorch == 0.4.1
```

## Structures

```python
# BPNN
BPNN(
  (embed): Embedding(59330, 50)
  (hid): Linear(in_features=250, out_features=300, bias=True)
  (out): Linear(in_features=300, out_features=35, bias=True)
  (dropout): Dropout(p=0.5)
)
# BPNN+CRF
BPNN(
  (embed): Embedding(59330, 50)
  (hid): Linear(in_features=250, out_features=300, bias=True)
  (out): Linear(in_features=300, out_features=35, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.5)
)
# BiLSTM+CRF
LSTM(
  (embed): Embedding(59330, 50)
  (lstm): LSTM(50, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=35, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.5)
)
# BiLSTM+CHAR+CRF
LSTM(
  (embed): Embedding(59330, 50)
  (clstm): CharLSTM(
    (embed): Embedding(59411, 50)
    (lstm): LSTM(50, 25, batch_first=True, bidirectional=True)
  )
  (wlstm): LSTM(100, 150, batch_first=True, bidirectional=True)
  (out): Linear(in_features=300, out_features=35, bias=True)
  (crf): CRF()
  (dropout): Dropout(p=0.5)
)
```

## Usage

```sh
usage: run.py [-h] [--crf] [--lstm] [--char] [--bidirectional] [--file FILE]
              [--threads THREADS]

Create Neural Network for POS Tagging.

optional arguments:
  -h, --help            show this help message and exit
  --crf                 use crf
  --lstm                use lstm
  --char                use char representation
  --bidirectional       use bidirectional lstm
  --file FILE, -f FILE  set where to store the model
  --threads THREADS, -t THREADS
                        set max num of threads
```

