# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from corpus import Corpus
from models import BPNN_CRF, LSTM_CRF, CHAR_LSTM_CRF

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create several models for POS Tagging.'
    )
    parser.add_argument('--model', '-m', default='char_lstm_crf',
                        choices=['bpnn_crf', 'lstm_crf', 'char_lstm_crf'],
                        help='choose the model for POS Tagging')
    parser.add_argument('--drop', action='store', default=0.5, type=float,
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
    trainset = corpus.load(config.ftrain, config.use_char, config.n_context)
    devset = corpus.load(config.fdev, config.use_char, config.n_context)
    testset = corpus.load(config.ftest, config.use_char, config.n_context)
    print(f"{'':2}size of trainset: {len(trainset)}\n"
          f"{'':2}size of devset: {len(devset)}\n"
          f"{'':2}size of testset: {len(testset)}\n")

    start = datetime.now()
    # 设置随机数种子
    torch.manual_seed(args.seed)

    print("Create Neural Network")
    if args.model == 'bpnn_crf':
        print(f"{'':2}n_context: {config.n_context}\n"
              f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        network = BPNN_CRF(n_context=config.n_context,
                           n_vocab=corpus.n_words,
                           n_embed=config.n_embed,
                           n_hidden=config.n_hidden,
                           n_out=corpus.n_tags,
                           embed=corpus.embed,
                           drop=args.drop)
    elif args.model == 'lstm_crf':
        print(f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        network = LSTM_CRF(n_vocab=corpus.n_words,
                           n_embed=config.n_embed,
                           n_hidden=config.n_hidden,
                           n_out=corpus.n_tags,
                           embed=corpus.embed,
                           drop=args.drop)
    elif args.model == 'char_lstm_crf':
        print(f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_char: {corpus.n_chars}\n"
              f"{'':2}n_char_embed: {config.n_char_embed}\n"
              f"{'':2}n_char_out: {config.n_char_out}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        network = CHAR_LSTM_CRF(n_vocab=corpus.n_words,
                                n_embed=config.n_embed,
                                n_char=corpus.n_chars,
                                n_char_embed=config.n_char_embed,
                                n_char_out=config.n_char_out,
                                n_hidden=config.n_hidden,
                                n_out=corpus.n_tags,
                                embed=corpus.embed,
                                drop=args.drop)
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
                test_loader=test_loader,
                epochs=args.epochs,
                interval=args.interval,
                eta=args.eta,
                file=args.file)

    # 载入训练好的模型
    network = torch.load(args.file)
    loss, accuracy = network.evaluate(test_loader)
    print(f"{'test:':<6} Loss: {loss:.4f} Accuracy: {accuracy:.2%}")
    print(f"{datetime.now() - start}s elapsed")
