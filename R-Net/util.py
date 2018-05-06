import tensorflow as tf
import re
from collections import Counter
import string
from bleu_metric.bleu import Bleu
from rouge_metric.rouge import Rouge


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores

def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        title_limit = config.title_limit

        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "title_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "is_selected" : tf.FixedLenFeature([], tf.int64),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })

        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        title_limit_idxs = tf.reshape(tf.decode_raw(
            features["title_idxs"], tf.int32), [title_limit])

        y1 = tf.reshape(tf.decode_raw(
            features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
            features["y2"], tf.float32), [para_limit])
        is_selected = features["is_selected"]
        qa_id = features["id"]
        return context_idxs, ques_idxs, title_limit_idxs, is_selected, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    ground_truths_dict = dict()
    prediction_dict = dict()
    for key, value in answer_dict.items():
        #因为这里只有一个答案，所以强制转换为list
        ground_truths = [eval_file[key]["answer"]]
        prediction = value
        ground_truths_dict[key] = ground_truths
        prediction_dict[key] = [prediction]
    mt_metric = compute_bleu_rouge(prediction_dict, ground_truths_dict)
    return mt_metric


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
