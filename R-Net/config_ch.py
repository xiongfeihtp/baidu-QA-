import os
import tensorflow as tf

from prepro_ch import prepro
from main import train, test

flags = tf.flags

train_file1 = '../preprocessed/trainset/search.train.json'
train_file2 = '../preprocessed/trainset/zhidao.train.json'

dev_file1 = '../preprocessed/devset/search.dev.json'
dev_file2 = '../preprocessed/devset/zhidao.dev.json'

test_file1 = '../test1set/preprocessed/search.test1.json'
test_file2 = '../test1set/preprocessed/zhidao.test1.json'
glove_word_file = '../cn2.vectors.bin'

target_dir = "data"
log_dir = "log/event"
save_dir = "log/model"
answer_dir = "log/answer"
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
word_count_file = os.path.join(target_dir, "word_count.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
answer_file = os.path.join(answer_dir, "answer.json")
#control the train process
flags.DEFINE_string("run_id","0","RUN ID[0]")
flags.DEFINE_string("model_name","basic","model name")
flags.DEFINE_string("load_path",None,"retrain_path")
flags.DEFINE_integer("load_step",0,"retrain globel step")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "prepro", "Running mode train/debug/test")
flags.DEFINE_string("word_count_file", word_count_file, "word_count_file")
flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file1 +' '+train_file2, "Train source file")
flags.DEFINE_string("dev_file", dev_file1 + ' '+ dev_file2, "Dev source file")
flags.DEFINE_string("test_file", test_file1 + ' '+ test_file2, "Test source file")
flags.DEFINE_string("glove_word_file", None , "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")

flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")

flags.DEFINE_integer("para_limit", 800, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn (should be False for CPU)")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
flags.DEFINE_float("keep_prob", 0.7, "Dropout keep prob in rnn")
flags.DEFINE_float("ptr_keep_prob", 0.7, "Dropout keep prob for pointer network")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 75, "Hidden size")
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")
flags.DEFINE_integer("title_limit", 30, "title limit for para")
flags.DEFINE_integer("max_to_keep", 10, "max number model for save")

def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        if config.use_cudnn:
            print("Warning: Due to a known bug in Tensorlfow, the parameters of CudnnGRU may not be properly restored.")
        test(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
