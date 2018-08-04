# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from corpus import Corpus
from model.bpnn import BPNN
from model.lstm import LSTM

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
    # 获取训练数据的句子
    sentences = Corpus.preprocess(config.ftrain)
    # 获取训练数据的所有不同词性
    words, tags = Corpus.parse(sentences)
    # 获取预训练词嵌入的词汇和嵌入矩阵
    words, embed = Corpus.get_embed(config.embed)
    # 以预训练词嵌入的词汇和训练数据的词性为基础建立语料
    corpus = Corpus(words, tags)

    print(f"\tsentences: {len(sentences)}\n"
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
    if args.lstm:
        pass
    else:
        print(f"\tvocdim: {corpus.nw}\n"
              f"\twindow: {config.window}\n"
              f"\tembdim: {config.embdim}\n"
              f"\thiddim: {config.hiddim}\n"
              f"\toutdim: {corpus.nt}")
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
                                  batch_size=len(testset))
    loss, tp, total, accuracy = network.evaluate(test_eval_loader)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
