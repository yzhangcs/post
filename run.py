# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F

from config import Config
from corpus import Corpus

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Neural Network for POS Tagging.'
    )
    parser.add_argument('--crf', action='store_true', default=False,
                        dest='crf', help='use crf')
    parser.add_argument('--lstm', action='store_true', default=False,
                        dest='lstm', help='use lstm')
    parser.add_argument('--char', action='store_true', default=False,
                        dest='char', help='use char representation')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        dest='bidirectional', help='use bidirectional lstm')
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
    config = Config(args.lstm)

    print("Preprocess the data")
    # 以训练数据为基础建立语料
    corpus = Corpus(config.ftrain)
    # 用预训练词嵌入扩展语料并返回词嵌入矩阵
    embed = corpus.extend(config.embed)
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
        from model.lstm import LSTM
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}window: {config.window}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n"
              f"{'':2}lossfn: {F.cross_entropy.__name__}\n")
        network = LSTM(vocdim=corpus.nw,
                       embdim=config.embdim,
                       window=config.window,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=F.cross_entropy,
                       use_crf=args.crf,
                       bidirectional=args.bidirectional,
                       pretrained=embed)
    elif args.lstm and args.char:
        from model.clstm import LSTM
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}chrdim: {corpus.nc}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}window: {config.window}\n"
              f"{'':2}cembdim: {config.cembdim}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n"
              f"{'':2}lossfn: {F.cross_entropy.__name__}\n")
        network = LSTM(vocdim=corpus.nw,
                       chrdim=corpus.nc,
                       embdim=config.embdim,
                       cembdim=config.cembdim,
                       window=config.window,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=F.cross_entropy,
                       use_crf=args.crf,
                       bidirectional=args.bidirectional,
                       pretrained=embed)
    else:
        from model.bpnn import BPNN
        print(f"{'':2}vocdim: {corpus.nw}\n"
              f"{'':2}embdim: {config.embdim}\n"
              f"{'':2}window: {config.window}\n"
              f"{'':2}hiddim: {config.hiddim}\n"
              f"{'':2}outdim: {corpus.nt}\n"
              f"{'':2}lossfn: {F.cross_entropy.__name__}\n")
        network = BPNN(vocdim=corpus.nw,
                       embdim=config.embdim,
                       window=config.window,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=F.cross_entropy,
                       use_crf=args.crf,
                       pretrained=embed)
    print(f"{network}\n")
    print("Use Adam optimizer to train the network")
    print(f"{'':2}epochs: {config.epochs}\n"
          f"{'':2}batch_size: {config.batch_size}\n"
          f"{'':2}interval: {config.interval}\n"
          f"{'':2}eta: {config.eta}\n"
          f"{'':2}lmbda: {config.lmbda}\n")
    network.fit(trainset, devset, file,
                epochs=config.epochs,
                batch_size=config.batch_size,
                interval=config.interval,
                eta=config.eta,
                lmbda=config.lmbda)

    # 载入训练好的模型
    network = torch.load(file)
    loss, tp, total, accuracy = network.evaluate(testset, config.batch_size)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
