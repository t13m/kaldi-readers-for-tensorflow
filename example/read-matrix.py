import sys
import tensorflow as tf
kaldi_module = tf.load_op_library("../../cmake-build-debug/libkaldi_readers.so")

def main():
    if len(sys.argv) != 2:
        print("Usage: python read-matrix.py /path/to/filename.scp")
        return 1
    scpfile = sys.argv[1]
    with open(scpfile) as fin:
        scplist = fin.readlines()
    scplist = [scpitem.strip().split()[1] for scpitem in scplist]
    value_rspecific = "./data/matrix.nocompress.ark:59"
    rspec = tf.placeholder(tf.string)
    feats_value = kaldi_module.read_and_decode_kaldi_matrix(rspec, left_padding=3, right_padding=4)
    #feats_value.set_shape([None, 4])
    feats_value.set_shape([None, None])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for rspec_value in scplist:
        feats = sess.run(feats_value, feed_dict={rspec: rspec_value})
        print(rspec_value)
        print(feats)


if __name__ == "__main__":
    main()
