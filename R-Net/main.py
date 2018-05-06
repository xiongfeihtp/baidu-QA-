import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from graph_handler import GraphHandler
from model import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat)
    graph_handler = GraphHandler(config,
                                 model)  # controls all tensors and variables in the graph, including loading /saving

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        graph_handler.initialize(sess)

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                handle: train_handle})

            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])

                graph_handler.add_summary(loss_sum, global_step)

            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                metrics_t, summs_t = evaluate_batch(
                    model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                graph_handler.add_summaries(summs_t, global_step)
                # save evaluation on train_batch
                train_file = "./log/train.txt"
                with open(train_file, 'a+') as f_train:
                    f_train.write(str(metrics_t) + ' ' + '\n')

                metrics, summs = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                graph_handler.add_summaries(summs, global_step)
                # save evaluation on dev_batch
                # divisor = dev_total // config.batch_size + 1
                train_file = "./log/dev.txt"
                with open(train_file, 'a+') as f_train:
                    f_train.write(str(metrics) + '\n')
                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                graph_handler.writer.flush()
                filename = os.path.join(
                    config.save_dir, "{}_{}.ckpt".format(config.model_name,global_step))
                graph_handler.save(sess, filename)

def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    summs = []
    for key in metrics:
        summs.append(tf.Summary(value=[tf.Summary.Value(
            tag="{}/{}".format(data_type, key), simple_value=metrics[key]), ]))
    return metrics, summs

def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
    total = meta["total"]
    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for _ in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        metrics['loss'] = loss
        print(metrics)
