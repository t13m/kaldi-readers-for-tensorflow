# Kaldi ark readers for tensorflow

## Introduction

This project aims to enable reading kaldi ark files into tensorflow. It adds
following operators to tensorflow:

* read_kaldi_matrix(rspecific)
* decode_kaldi_matrix(data, element-type)
* read_kaldi_post_and_ali(rspecific, is_reading_post(bool))
* decode_kaldi_ali(data, element-type, is_reading_post(bool))

For kaldi matrix, only float matrix are supported. Please pass "tf.float32" in
through the element-type argument.

For compressed kaldi matrix, only compression method 2 (kSpeechFeature) is supported.

For alignment, user need to specify weather is reading posteriors or pdfs by the argument
`is_reading_post`. Operator decode_kaldi_ali produces alignment pdfs output, in format
of a one dimension int32 tensor (a int32 tensor). Please pass "tf.int32" in through the
element-type argument.

Contributions are welcome. Feel free to fork and send pull request or to create issues.

## Build

Prerequisitions:

* Linux
* GCC version > 5.1 (for use of c++11 regex)
* python with tensorflow installed

Steps:

1. git clone <url-of-this-repo>
2. cd kaldi-reader-standalone
3. mkdir build && cd build
4. cmake .. -DPYTHONBIN=/path/to/your/correct/version/of/python
5. make

Then you are all set.

## Usage example

```python
    kaldi_module = kaldi_module = tf.load_op_library("/path/to/this/project/libkaldi_readers.so")
    feats_raw_value = kaldi_module.read_kaldi_matrix("/path/to/somearks/file1.ark:2321")
    feats_value = kaldi_module.decode_kaldi_matrix(feats_raw_value, tf.float32)
    feats_value.set_shape([None, num_dim])
```

There are some examples under the `example` directory. To run them, please modify the library path (in contents of the
python files) to the correct path.

1. cd example
2. python read-compressed-matrix.py
3. python read-uncompressed-matrix.py
4. python read-post.py
5. python read-ali.py

## Author

Fan Ziye

## Reference

Kaldi: https://github.com/kaldi-asr/kaldi
Tensorflow: https://www.tensorflow.org/extend/adding_an_op