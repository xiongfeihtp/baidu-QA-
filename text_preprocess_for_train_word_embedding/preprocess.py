'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: preprocess.py
@time: 2018/4/24 下午7:32
@desc: shanghaijiaotong university
'''
import os
import tensorflow as tf
import jieba
import json
from gensim import corpora, models
import re
from tqdm import tqdm
from itertools import chain



flags = tf.flags

train_file1 = '../preprocessed/trainset/search.train.json'
train_file2 = '../preprocessed/trainset/zhidao.train.json'

dev_file1 = '../preprocessed/devset/search.dev.json'
dev_file2 = '../preprocessed/devset/zhidao.dev.json'

test_file1 = '../test1set/preprocessed/search.test1.json'
test_file2 = '../test1set/preprocessed/zhidao.test1.json'

target_dir = "data"

flags.DEFINE_string("train_file", train_file1 +' '+train_file2, "Train source file")
flags.DEFINE_string("dev_file", dev_file1 + ' '+ dev_file2, "Dev source file")
flags.DEFINE_string("test_file", test_file1 + ' '+ test_file2, "Test source file")


def get_stop_dict(stop_file):
    stop_words = {}
    fstop = open(stop_file, 'r')
    for item in fstop:
        stop_words[item.strip()] = item.strip()
    fstop.close()
    return stop_words

def text_generate_with_label(filename, mode = 'train'):
    filename = filename.split(' ')
    print("preprocess: ", filename)
    total = 0
    with open(filename[0], 'r') as fs, open(filename[1], 'r') as fz:
        fp = open("cut_result_{}.txt".format(mode), 'w')
        for line in chain(fs.readlines(), fz.readlines()):
            total += 1
            if not (total % 10000):
                print(total)
            article = json.loads(line)
            data_list = []
            ques = article['segmented_question']
            data_list.append(ques)
            try:
                answers = article['segmented_answers']  # list
                for answer in answers:
                    data_list.append(answer)
                fake_answer = article['fake_answers']
                fake_answer = list(jieba.cut(fake_answer[0]))
                data_list.append(fake_answer)
            except Exception as e:
                pass
            for i, para in enumerate(article['documents']):
                contexts = para["segmented_paragraphs"]  # list
                for context in contexts:
                    data_list.append(context)
                title = para["title"] #string
                title = list(jieba.cut(title))
                data_list.append(title)
            for list_item in data_list:
                print(' '.join(list_item), file = fp)
        fp.close()

def text_generate_without_label(filename, mode = 'test'):
    filename = filename.split(' ')
    print("preprocess: ", filename)
    total = 0
    with open(filename[0], 'r') as fs, open(filename[1], 'r') as fz:
        fp = open("cut_result_{}.txt".format(mode), 'w')
        for line in chain(fs.readlines(), fz.readlines()):
            total += 1
            if  not (total % 10000):
                print(total)
            article = json.loads(line)
            data_list = []
            ques = article['segmented_question']
            data_list.extend(ques)
            for i, para in enumerate(article['documents']):
                context = para["segmented_paragraphs"]  # list
                data_list.extend(context)
                title = para["title"] #string
                title = list(jieba.cut(title))
                data_list.extend(title)
            for list_item in data_list:
                print(' '.join(list_item), file=fp)
        fp.close()

def main(_):
    config = flags.FLAGS
    text_generate_with_label(config.train_file, "train")
    text_generate_with_label(config.dev_file, "dev")
    text_generate_without_label(config.test_file, 'test')
    #
    # fr = open('cut_result.txt', 'r')
    # train = []
    # for line in fr.readlines():
    #     # 读写数据存在多余的空格行
    #     if line != '\n':
    #         line = line.split(' ')
    #         train.append(line)
    # fr.close()
    # dictionary = corpora.Dictionary(train)
    # dictionary.save_as_text('dictionary.txt')
    # dictionary.save_as_text('dictionary_freq.txt')

if __name__ == "__main__":
    tf.app.run()
