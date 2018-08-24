# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from corpus import Corpus
from models import ATTN, BPNN, LSTM, LSTM_CHAR

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Neural Network for POS Tagging.'
    )
    parser.add_argument('--crf', action='store_true', default=False,
                        dest='crf', help='use crf')
    parser.add_argument('--attn', action='store_true', default=False,
                        dest='attn', help='use attention')
    parser.add_argument('--bpnn', action='store_true', default=False,
                        dest='bpnn', help='use bpnn')
    parser.add_argument('--lstm', action='store_true', default=False,
                        dest='lstm', help='use lstm')
    parser.add_argument('--char', action='store_true', default=False,
                        dest='char', help='use char representation')
    parser.add_argument('--bi', action='store_true', default=False,
                        dest='bi', help='use bidirectional lstm')
    parser.add_argument('--embed', '-e', action='store_true', default=False,
                        dest='embed', help='use pretrained embedding file')
    parser.add_argument('--file', '-f', action='store', dest='file',
                        help='set where to store the model')
    parser.add_argument('--threads', '-t', action='store', dest='threads',
                        default='4', type=int, help='set max num of threads')
    args = parser.parse_args()

    # 设置随机数种子
    torch.manual_seed(1)
    # 设置最大线程数
    torch.set_num_threads(args.threads)
    print(f"Set max num of threads to {args.threads}")

    # 根据参数读取配置
    config = Config(args.bpnn)

    print("Preprocess the data")
    # 以训练数据为基础建立语料
    corpus = Corpus(config.ftrain)
    # 用预训练词嵌入扩展语料并返回词嵌入矩阵
    embed = corpus.extend(config.embed) if args.embed else None
    print(corpus)

    print("Load the dataset")
    trainset = corpus.load(config.ftrain, args.char, config.window)
    devset = corpus.load(config.fdev, args.char, config.window)
    testset = corpus.load(config.ftest, args.char, config.window)
    print(f"{'':2}size of trainset: {len(trainset)}\n"
          f"{'':2}size of devset: {len(devset)}\n"
          f"{'':2}size of testset: {len(testset)}\n")
    file = args.file if args.file else config.netpt

    start = datetime.now()

    print("Create Neural Network")
    if args.lstm and not args.char:
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = LSTM(vocdim=corpus.nw,
                       embdim=config.embdim,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=nn.CrossEntropyLoss(),
                       use_crf=args.crf,
                       bidirectional=args.bi,
                       embed=embed)
    elif args.lstm and args.char:
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}chrdim: {corpus.nc}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}char_hiddim: {config.char_hiddim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = LSTM_CHAR(vocdim=corpus.nw,
                            chrdim=corpus.nc,
                            embdim=config.embdim,
                            char_hiddim=config.char_hiddim,
                            hiddim=config.hiddim,
                            outdim=corpus.nt,
                            lossfn=nn.CrossEntropyLoss(),
                            use_attn=args.attn,
                            use_crf=args.crf,
                            bidirectional=args.bi,
                            embed=embed)
    elif args.attn:
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}outdim: {corpus.nt}\n")
        network = ATTN(vocdim=corpus.nw,
                       embdim=config.embdim,
                       outdim=corpus.nt,
                       lossfn=nn.CrossEntropyLoss(),
                       use_crf=args.crf,
                       embed=embed)
    else:
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
                       use_crf=args.crf,
                       embed=embed)
    print(f"{network}\n")

    # 设置数据加载器
    train_loader = DataLoader(dataset=trainset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=network.collate_fn)
    dev_loader = DataLoader(dataset=devset,
                            batch_size=config.batch_size,
                            collate_fn=network.collate_fn)
    test_loader = DataLoader(dataset=devset,
                             batch_size=config.batch_size,
                             collate_fn=network.collate_fn)

    print("Use Adam optimizer to train the network")
    print(f"{'':2}epochs: {config.epochs}\n"
          f"{'':2}batch_size: {config.batch_size}\n"
          f"{'':2}interval: {config.interval}\n"
          f"{'':2}eta: {config.eta}\n")
    network.fit(train_loader=train_loader,
                dev_loader=dev_loader,
                epochs=config.epochs,
                interval=config.interval,
                eta=config.eta,
                file=file)

    # 载入训练好的模型
    network = torch.load(file)
    loss, tp, total, accuracy = network.evaluate(loader)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
