import tensorflow as tf
import random
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np
from itertools import chain

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def get_match_score(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    score = len(set1 & set2) / len(set1 | set2)
    return score


def process_file_label_data(filename, data_type, word_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    filename = filename.split(' ')
    with open(filename[0], "r") as fs, open(filename[1], 'r') as fz:
        for line in chain(fs.readlines(), fz.readlines()):
            article = json.loads(line)
            #部分样本不存在答案
            ques_tokens = article['segmented_question']
            answer_tokens = article['segmented_answers']  # list
            if answer_tokens == []:
                continue
            all_answer_tokens = []
            for answer_token in answer_tokens:
                all_answer_tokens.extend(answer_token)

            # answer
            for token in all_answer_tokens:
                word_counter[token] += 1
            del all_answer_tokens
            # question
            for token in ques_tokens:
                word_counter[token] += 1

            answers = article['answers']  # list

            # fake_answer_span
            answer_span = article['answer_spans']
            fake_answer = article['fake_answers']
            answer_docs = article['answer_docs']

            uuid = article['question_id']
            for i, para in enumerate(article["documents"]):
                #记录进度
                total += 1
                if not (total % 10000):
                    print(total)
                context_tokens = para["segmented_paragraphs"]  # list
                all_context_tokens = []
                for context_token in context_tokens:
                    all_context_tokens.extend(context_token)
                all_context = []
                for context_token in context_tokens:
                    all_context.append(''.join(context_token))
                all_context = ''.join(all_context)
                #context = para["paragraphs"]  # list
                #all_context = ''.join(context)
                title_tokens = para["segmented_title"]
                is_selected = para["is_selected"]
                spans = convert_idx(all_context, all_context_tokens)
                # document
                for token in all_context_tokens:
                    word_counter[token] += 1
                # title
                for token in title_tokens:
                    word_counter[token] += 1

                # fake answer
                y1 = 0
                y2 = 0
                para_num = para['most_related_para']
                is_answer_docs = 0
                if i == answer_docs:
                    is_answer_docs = 1
                    before_len = 0
                    count = 0
                    while (count < para_num):
                        print(count, para_num)
                        before_len += len(context_tokens[count])
                        count += 1
                    y1 = answer_span[0] + before_len
                    y2 = answer_span[1] + before_len

                # combine answer
                # m_answer
                m_y1 = 0
                m_y2 = 0
                match_answer = ""
                if is_selected:
                    selected_param_tokens = context_tokens[para_num]
                    match_score_dict = {}
                    for i, answer_token in enumerate(answer_tokens):
                        match_score_dict[i] = get_match_score(answer_token, selected_param_tokens)
                    sorted_score = sorted(match_score_dict.items(), key=lambda item: item[1], reverse=True)
                    match_answer = answers[sorted_score[0][0]]
                    count = 0
                    before_len = 0
                    while count < para_num:
                        before_len += len(context_tokens[count])
                        count += 1

                    m_y1 = before_len
                    m_y2 = before_len + len(selected_param_tokens)

                example = {"span": spans, "title_tokens": title_tokens, "context_tokens": all_context_tokens,
                           "y1": y1, "y2": y2, "m_y1": m_y1, "m_y2": m_y2,
                           "ques_tokens": ques_tokens, "id": total, 'is_selected': is_selected,
                           "is_answer_doc": is_answer_docs}

                examples.append(example)
                eval_examples[str(total)] = {"context": all_context, "spans": spans,
                                             "fake_answers": fake_answer, "answer": match_answer,
                                             "uuid": uuid, "is_answer_doc": is_answer_docs}

        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        return examples, eval_examples


def process_file_without_label(filename, data_type, word_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    filename = filename.split(' ')
    with open(filename[0], "r") as fs, open(filename[1], 'r') as fz:
        for line in chain(fs.readlines(), fz.readlines()):
            article = json.loads(line)
            ques_tokens = article['segmented_question']

            # question
            for token in ques_tokens:
                word_counter[token] += 1
            id = article['question_id']
            for i, para in enumerate(article["documents"]):
                if not (total % 10000):
                    print(total)
                total += 1
                context_tokens = para["segmented_paragraphs"]  # list
                all_context_tokens = []
                all_context = []
                for context_token in context_tokens:
                    all_context_tokens.extend(context_token)
                for context_token in context_tokens:
                    all_context.append(''.join(context_token))
                #context = para["paragraphs"]  # list
                #all_context = ''.join(context)
                all_context = ''.join(all_context)
                title_tokens = para["segmented_title"]
                try:
                    spans = convert_idx(all_context, all_context_tokens)
                except Exception as e:
                    print(para)
                    raise Exception
                # document
                for token in all_context_tokens:
                    word_counter[token] += 1
                # title
                for token in title_tokens:
                    word_counter[token] += 1

                # fake answer
                y1 = 0
                y2 = 0
                # combine answer
                # m_answer
                m_y1 = 0
                m_y2 = 0

                example = {"span": spans, "title_tokens": title_tokens, "context_tokens": all_context_tokens,
                           "y1": y1, "y2": y2, "m_y1": m_y1, "m_y2": m_y2,
                           "ques_tokens": ques_tokens, "id": total, 'is_selected': 0}
                examples.append(example)
                eval_examples[str(total)] = {"context": all_context, "spans": spans, "uuid": id, 'id': total}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
        return examples, eval_examples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0])
                vector = list(map(float, array[1:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                                     token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def build_features(config, examples, data_type, out_file, word2idx_dict, is_test=False):
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    title_limit = config.title_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit or len(
            example["title_tokens"]) > title_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}

    for example in tqdm(examples):
        total_ += 1
        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        title_idxs = np.zeros([title_limit], dtype=np.int32)

        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        m_y1 = np.zeros([para_limit], dtype=np.float32)
        m_y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in word:
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["title_tokens"]):
            title_idxs[i] = _get_word(token)

        start, end = example["m_y1"], example["m_y2"]
        if start == 0 and end == 0:
            pass
        else:
            y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "title_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[title_idxs.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "is_selected": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["is_selected"]])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta

def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def prepro(config):
    word_counter = Counter()
    train_examples, train_eval = process_file_label_data(
        config.train_file, "train", word_counter)

    dev_examples, dev_eval = process_file_label_data(
        config.dev_file, "dev", word_counter)

    test_examples, test_eval = process_file_without_label(
        config.test_file, "test", word_counter)

    word_emb_file = config.glove_word_file

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, is_test=True)

    save(config.word_count_file, word_counter, message="word_count")
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
