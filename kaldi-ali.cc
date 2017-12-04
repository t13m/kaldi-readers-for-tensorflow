#include <memory>
#include <regex>
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

#include "shape-funcs.hh"

namespace tensorflow {
    using shape_util::ScalarInputsAndOutputs;
    using shape_util::TwoElementOutput;

    static Status ReadKaldiPostAndAli(Env* env, const string& ark_path, uint64 ark_offset, bool is_reading_post, string* contents) {
        enum { kBufferSize = 256 << 10 /* 256 kB */ };

        std::unique_ptr<RandomAccessFile> file_;
        std::unique_ptr<io::InputStreamInterface> buffered_inputstream_;

        TF_RETURN_IF_ERROR(env->NewRandomAccessFile(ark_path, &file_));
        buffered_inputstream_.reset(
                new io::BufferedInputStream(file_.get(), kBufferSize));
        TF_RETURN_IF_ERROR(buffered_inputstream_->SkipNBytes(ark_offset));

        // Actural reading start from here
        string binary;
        TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(2, &binary));
        CHECK_EQ(binary[0], '\0');
        CHECK_EQ(binary[1], 'B');
        string header_buffer;
        TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(1, &header_buffer));
        if (header_buffer[0] == '\4') {
            // This is a vector of int
            string size_str;
            buffered_inputstream_->ReadNBytes(4, &size_str);
            int32 size = *reinterpret_cast<const int32*>(size_str.data());
            string data;
            if (is_reading_post) {
                for (int32 outer_vec_idx = 0; outer_vec_idx < size; outer_vec_idx++) {
                    // <1> <4> [<1> <4> <1> <4>] [<1> <4> <1> <4>]
                    string inner_size_str;
                    buffered_inputstream_->ReadNBytes(5, &inner_size_str);
                    int32 inner_size = *reinterpret_cast<const int32 *>(inner_size_str.data() + 1);
                    string inner_vec_data;
                    buffered_inputstream_->ReadNBytes(inner_size * 10, &inner_vec_data);
                    data += inner_size_str + inner_vec_data;
                }
            } else {
                TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(size * 5, &data));
            }
            *contents = header_buffer + size_str + data;
        } else {
            return Status(error::UNAVAILABLE, "Unknown Kaldi Post or Ali: " + header_buffer);
        }
        return Status::OK();
    }

    class ReadKaldiPostAndAliOp : public OpKernel {
    public:
        using OpKernel::OpKernel;
        explicit ReadKaldiPostAndAliOp(OpKernelConstruction *context)
                :OpKernel(context),
                 id_pat_("^(\\S+):(\\d+)")
        {
            OP_REQUIRES_OK(context, context->GetAttr("is_reading_post", &is_reading_post_));
        }
        void Compute(OpKernelContext* context) override {

            const Tensor* input;

            OP_REQUIRES_OK(context, context->input("scpline", &input));
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                        errors::InvalidArgument(
                                "Input filename tensor must be scalar, but had shape: ",
                                input->shape().DebugString()));

            Tensor* output = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output("contents",
                                                             TensorShape({}), &output));
            const std::regex id_pat("^(\\S+):(\\d+)");
            std::smatch m;
            string half_scp_line = input->scalar<string>()();
            bool matched = std::regex_search(half_scp_line, m, id_pat);
            OP_REQUIRES(context, matched, Status(error::INVALID_ARGUMENT, "Script line is " + half_scp_line));
            string ark_path = m[1];
            string ark_offset_str = m[2];
            uint64 ark_offset = std::stoull(ark_offset_str);

            OP_REQUIRES_OK(context,
                           ReadKaldiPostAndAli(context->env(), ark_path, ark_offset, is_reading_post_,
                                               &output->scalar<string>()()));
        }
    private:
        bool is_reading_post_;
        const std::regex id_pat_;
    };
    REGISTER_KERNEL_BUILDER(Name("ReadKaldiPostAndAli").Device(DEVICE_CPU), ReadKaldiPostAndAliOp);

    REGISTER_OP("ReadKaldiPostAndAli")
            .Attr("is_reading_post: bool")
            .Input("scpline: string")
            .Output("contents: string")
            .SetShapeFn(ScalarInputsAndOutputs)
            .Doc(R"doc(
Reads and outputs the entire contents of the input kaldi post or ali ark filename.

scpline: scalar. /path/to/ark.file:12345
)doc");

    class DecodeKaldiAliOp : public OpKernel {
    public:
        explicit DecodeKaldiAliOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
            OP_REQUIRES_OK(context, context->GetAttr("is_reading_post", &is_reading_post_));
        }

        void Compute(OpKernelContext* context) override {
            const auto& input = context->input(0);
            int64 str_size = -1;
            auto flat_in = input.flat<string>();
            OP_REQUIRES(context, flat_in.size() == 1,
                        errors::InvalidArgument(
                                "DecodeKaldiAliOp requires input string size = 1"
                        )
            )
            const string& in_str = flat_in(0);
            str_size = in_str.size();

            const char* in_data = reinterpret_cast<const char*>(flat_in(0).data());
            TensorShape out_shape;
            int32 num_elem = *reinterpret_cast<const int32*>(in_data + 1);
            out_shape.AddDim(num_elem);

            if (str_size == -1 || str_size == 0) {  // Empty input
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output("output", out_shape,
                                                                 &output_tensor));
                return;
            }

            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(
                    context, context->allocate_output("output", out_shape, &output_tensor));
            auto out = output_tensor->flat<int32>();

            int32* out_data = out.data();
            const char* in_bytes = in_data + 5;
            if (is_reading_post_) {
                for (int32 frame_idx = 0; frame_idx < num_elem; frame_idx++) {
                    out_data[frame_idx] = *reinterpret_cast<const int32*>(in_bytes + 5 + 1);
                    in_bytes += 15;
                }
            } else {
                for (int32 frame_idx = 0; frame_idx < num_elem; frame_idx++) {
                    out_data[frame_idx] = *reinterpret_cast<const int32*>(in_bytes + 1);
                    in_bytes += 5;
                }
            }
        }

    private:
        bool is_reading_post_;
        DataType out_type_;

    };

    REGISTER_KERNEL_BUILDER(Name("DecodeKaldiAli").Device(DEVICE_CPU), DecodeKaldiAliOp);

    REGISTER_OP("DecodeKaldiAli")
            .Input("bytes: string")
            .Output("output: out_type")
            .Attr("out_type: {int32}")
            .Attr("is_reading_post: bool")
            .SetShapeFn(shape_inference::UnknownShape)
            .Doc(R"doc(
Reinterpret the bytes of a string as a kaldi ali
)doc");


}  // namespace tensorflow
