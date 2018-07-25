# -*- coding: utf-8 -*-

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from corpus import Corpus
from network import Network

# 解析命令参数
parser = argparse.ArgumentParser(
    description='Create Neural Network for POS Tagging.'
)
parser.add_argument('--file', '-f', action='store', dest='file',
                    help='set where to store the model')
parser.add_argument('--threads', '-t', action='store', dest='threads',
                    default='4', type=int, help='set the max num of threads')
args = parser.parse_args()

# 设置最大线程数
torch.set_num_threads(args.threads)


if __name__ == '__main__':
    # 根据参数读取配置
    config = Config()

    print("Preprocessing the data")
    corpus = Corpus(config.ftrain)
    print("\tsentences: {}\n"
          "\tdifferent words: {}\n"
          "\tdifferent tags: {}".format(len(corpus.sentences), *corpus.size()))
    train_data = corpus.load(config.ftrain, window=config.window)
    dev_data = corpus.load(config.fdev, window=config.window)
    test_data = corpus.load(config.ftest, window=config.window)
    embed = corpus.get_embed(config.embed)
    file = args.file if args.file else config.netpkl

    start = datetime.now()
    print("Creating Artificial Neural Network")
    network = Network(vocdim=corpus.nw,
                      window=config.window,
                      embdim=config.embdim,
                      hiddim=config.hiddim,
                      outdim=corpus.nt,
                      lossfn=F.cross_entropy,
                      embed=embed)
    print("Using Adam optimizer to train the network")
    print(f"\tsize of train_data: {len(train_data[0])}\n"
          f"\tsize of dev_data: {len(dev_data[0])}\n"
          f"\tmax num of threads: {args.threads}\n")
    print(f"\tepochs: {config.epochs}\n"
          f"\tbatch_size: {config.batch_size}\n"
          f"\tinterval: {config.interval}\n"
          f"\teta: {config.eta}\n"
          f"\tlmbda: {config.lmbda}\n")
    network.train(train_data, dev_data, file,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  interval=config.interval,
                  eta=config.eta,
                  lmbda=config.lmbda)
    loss, tp, total, accuracy = network.evaluate(test_data)
    print(f"Loss of test: {loss:.4f}")
    print(f"Accuracy of test: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
