import tensorflow as tf
kaldi_module = tf.load_op_library("../build/libkaldi_readers.so")

def main():
    value_rspecific = "./data/ali.ark:6"
    rspec = tf.constant(value_rspecific, tf.string)
    ali_raw_value = kaldi_module.read_kaldi_post_and_ali(rspec, is_reading_post=False)
    ali_value = kaldi_module.decode_kaldi_ali(ali_raw_value, tf.int32, is_reading_post=False)
    ali_value.set_shape([None])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ali = sess.run(ali_value)
    print(ali.shape)
    print(ali)


if __name__ == "__main__":
    main()
