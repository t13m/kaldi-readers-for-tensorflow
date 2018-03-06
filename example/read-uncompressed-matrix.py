import tensorflow as tf
kaldi_module = tf.load_op_library("../build/libkaldi_readers.so")

def main():
    value_rspecific = "./data/matrix.nocompress.ark:59"
    rspec = tf.constant(value_rspecific, tf.string)
    feats_value = kaldi_module.read_and_decode_kaldi_matrix(rspec, left_padding=3, right_padding=4)
    feats_value.set_shape([None, 4])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feats = sess.run(feats_value)
    print(feats.shape)
    print(feats)


if __name__ == "__main__":
    main()
