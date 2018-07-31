# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from corpus import Corpus
from net import Network

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
    # 获取预训练词嵌入的词汇和嵌入矩阵
    words, embed = Corpus.get_embed(config.embed)
    # 获取训练数据的句子
    sentences = Corpus.preprocess(config.ftrain)
    # 获取训练数据的所有不同词性
    tags = Corpus.parse(sentences)[1]
    # 以预训练词嵌入的词汇和训练数据的词性为基础建立语料
    corpus = Corpus(words, tags)

    print("\tsentences: {}\n"
          "\tdifferent words: {}\n"
          "\tdifferent tags: {}".format(len(sentences), *corpus.size()))
    # 获取数据
    train_data = corpus.load(config.ftrain, config.window)
    dev_data = corpus.load(config.fdev, config.window)
    test_data = corpus.load(config.ftest, config.window)
    file = args.file if args.file else config.netpkl

    start = datetime.now()
    print("Creating Neural Network")
    print(f"\tvocdim: {corpus.nw}\n"
          f"\twindow: {config.window}\n"
          f"\tembdim: {config.embdim}\n"
          f"\thiddim: {config.hiddim}\n"
          f"\toutdim: {corpus.nt}\n")
    net = Network(vocdim=corpus.nw,
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
    net.fit(train_data, dev_data, file,
            epochs=config.epochs,
            batch_size=config.batch_size,
            interval=config.interval,
            eta=config.eta,
            lmbda=config.lmbda)

    # 载入训练好的模型
    net = Network.load(file)
    loss, tp, total, accuracy = net.evaluate(test_data)
    print(f"{'test:':<6} "
          f"Loss: {loss:.4f} "
          f"Accuracy: {tp} / {total} = {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed\n")
