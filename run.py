# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from corpus import Corpus
from models import BPNN, LSTM, LSTM_CHAR

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create several models for POS Tagging.'
    )
    parser.add_argument('--model', '-m', default='lstm_char',
                        choices=['bpnn', 'lstm', 'lstm_char'],
                        help='choose the model for POS Tagging')
    parser.add_argument('--crf', action='store_true', default=False,
                        help='use crf')
    parser.add_argument('--prob', action='store', default=0.5, type=float,
                        help='set the prob of dropout')
    parser.add_argument('--batch_size', action='store', default=50, type=int,
                        help='set the size of batch')
    parser.add_argument('--epochs', action='store', default=100, type=int,
                        help='set the max num of epochs')
    parser.add_argument('--interval', action='store', default=10, type=int,
                        help='set the max interval to stop')
    parser.add_argument('--eta', action='store', default=0.001, type=float,
                        help='set the learning rate of training')
    parser.add_argument('--threads', '-t', action='store', default=4, type=int,
                        help='set the max num of threads')
    parser.add_argument('--seed', '-s', action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--file', '-f', action='store', default='network.pt',
                        help='set where to store the model')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}\n"
          f"Set the seed for generating random numbers to {args.seed}\n")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    # 根据模型读取配置
    config = config.config[args.model]

    print("Preprocess the data")
    # 建立语料
    corpus = Corpus(config.ftrain, config.fembed)
    print(corpus)

    print("Load the dataset")
    trainset = corpus.load(config.ftrain, config.charwise, config.window)
    devset = corpus.load(config.fdev, config.charwise, config.window)
    testset = corpus.load(config.ftest, config.charwise, config.window)
    print(f"{'':2}size of trainset: {len(trainset)}\n"
          f"{'':2}size of devset: {len(devset)}\n"
          f"{'':2}size of testset: {len(testset)}\n")

    start = datetime.now()
    # 设置随机数种子
    torch.manual_seed(args.seed)

    print("Create Neural Network")
    if args.model == 'bpnn':
        print(f"{'':2}window: {config.window}\n"
              f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = BPNN(window=config.window,
                       vocdim=corpus.nw,
                       embdim=config.embdim,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=nn.CrossEntropyLoss(),
                       embed=corpus.embed,
                       crf=args.crf,
                       p=args.prob)
    elif args.model == 'lstm':
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = LSTM(vocdim=corpus.nw,
                       embdim=config.embdim,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=nn.CrossEntropyLoss(),
                       embed=corpus.embed,
                       crf=args.crf,
                       p=args.prob)
    elif args.model == 'lstm_char':
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}chrdim: {corpus.nc}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}char_embdim: {config.char_embdim}\n"
              f"{'':2}char_outdim: {config.char_outdim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = LSTM_CHAR(vocdim=corpus.nw,
                            chrdim=corpus.nc,
                            embdim=config.embdim,
                            char_embdim=config.char_embdim,
                            char_outdim=config.char_outdim,
                            hiddim=config.hiddim,
                            outdim=corpus.nt,
                            lossfn=nn.CrossEntropyLoss(),
                            embed=corpus.embed,
                            crf=args.crf,
                            p=args.prob)
    print(f"{network}\n")

    # 设置数据加载器
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=network.collate_fn)
    dev_loader = DataLoader(dataset=devset,
                            batch_size=args.batch_size,
                            collate_fn=network.collate_fn)
    test_loader = DataLoader(dataset=testset,
                             batch_size=args.batch_size,
                             collate_fn=network.collate_fn)

    print("Use Adam optimizer to train the network")
    print(f"{'':2}epochs: {args.epochs}\n"
          f"{'':2}batch_size: {args.batch_size}\n"
          f"{'':2}interval: {args.interval}\n"
          f"{'':2}eta: {args.eta}\n")
    network.fit(train_loader=train_loader,
                dev_loader=dev_loader,
                epochs=args.epochs,
                interval=args.interval,
                eta=args.eta,
                file=args.file)

    # 载入训练好的模型
    network = torch.load(args.file)
    loss, tp, total, accuracy = network.evaluate(test_loader)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed")
