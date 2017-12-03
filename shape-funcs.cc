
#include "shape-funcs.hh"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace shape_util {
    using tensorflow::shape_inference::DimensionHandle;
    using tensorflow::shape_inference::InferenceContext;
    using tensorflow::shape_inference::ShapeHandle;

    tensorflow::Status ScalarInputsAndOutputs(InferenceContext *c) {
        ShapeHandle unused;
        for (int i = 0; i < c->num_inputs(); ++i) {
            TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
        }
        for (int i = 0; i < c->num_outputs(); ++i) {
            c->set_output(i, c->Scalar());
        }
        return tensorflow::Status::OK();
    }

    tensorflow::Status TwoElementOutput(InferenceContext *c) {
        c->set_output(0, c->Vector(2));
        return tensorflow::Status::OK();
    }
} // namespace shape_util