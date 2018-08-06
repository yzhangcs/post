# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

    print(f"\tsentences: {corpus.ns}\n"
          f"\tdifferent words: {corpus.nw - 3}\n"
          f"\tdifferent tags: {corpus.nt}")

    # 获取数据
    train_data = corpus.load(config.ftrain, args.char, config.window)
    dev_data = corpus.load(config.fdev, args.char, config.window)
    test_data = corpus.load(config.ftest, args.char, config.window)
    print(f"\tsize of train_data: {len(train_data[0])}\n"
          f"\tsize of dev_data: {len(dev_data[0])}")
    file = args.file if args.file else config.netpkl

    start = datetime.now()

    print("Create Neural Network")
    if args.lstm and not args.char:
        from model.lstm import LSTM
        print(f"\tvocdim: {corpus.nw}\n"
              f"\tembdim: {config.embdim}\n"
              f"\twindow: {config.window}\n"
              f"\thiddim: {config.hiddim}\n"
              f"\toutdim: {corpus.nt}\n"
              f"\tlossfn: {F.cross_entropy.__name__}")
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
        print(f"\tvocdim: {corpus.nw}\n"
              f"\tchrdim: {corpus.nc}\n"
              f"\tembdim: {config.embdim}\n"
              f"\twindow: {config.window}\n"
              f"\tcembdim: {config.cembdim}\n"
              f"\thiddim: {config.hiddim}\n"
              f"\toutdim: {corpus.nt}\n"
              f"\tlossfn: {F.cross_entropy.__name__}")
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
        print(f"\tvocdim: {corpus.nw}\n"
              f"\tembdim: {config.embdim}\n"
              f"\twindow: {config.window}\n"
              f"\thiddim: {config.hiddim}\n"
              f"\toutdim: {corpus.nt}\n"
              f"\tlossfn: {F.cross_entropy.__name__}")
        network = BPNN(vocdim=corpus.nw,
                       embdim=config.embdim,
                       window=config.window,
                       hiddim=config.hiddim,
                       outdim=corpus.nt,
                       lossfn=F.cross_entropy,
                       use_crf=args.crf,
                       pretrained=embed)
    print(network)
    print("Use Adam optimizer to train the network")
    print(f"\tepochs: {config.epochs}\n"
          f"\tbatch_size: {config.batch_size}\n"
          f"\tinterval: {config.interval}\n"
          f"\teta: {config.eta}\n"
          f"\tlmbda: {config.lmbda}\n")
    network.fit(train_data, dev_data, file,
                epochs=config.epochs,
                batch_size=config.batch_size,
                interval=config.interval,
                eta=config.eta,
                lmbda=config.lmbda)

    # 载入训练好的模型
    network = torch.load(file)
    testset = TensorDataset(*test_data)
    test_eval_loader = DataLoader(dataset=testset,
                                  batch_size=len(testset),
                                  collate_fn=network.collate_fn)
    loss, tp, total, accuracy = network.evaluate(test_eval_loader)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
