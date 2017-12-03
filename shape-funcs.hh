//
// Created by zyfan on 12/3/17.
//

#ifndef KALDI_READER_STANDALONE_SHAPE_FUNCS_HH
#define KALDI_READER_STANDALONE_SHAPE_FUNCS_HH
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace shape_util {
    using tensorflow::shape_inference::InferenceContext;

    tensorflow::Status ScalarInputsAndOutputs(InferenceContext *c);

    tensorflow::Status TwoElementOutput(InferenceContext *c);
} // namespace shape_util


#endif //KALDI_READER_STANDALONE_SHAPE_FUNCS_HH
