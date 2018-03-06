/* Reference:
 * */

#include <memory>
#include <regex>
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "shape-funcs.hh"

namespace tensorflow {
    using shape_util::ScalarInputsAndOutputs;
    using shape_util::TwoElementOutput;

    static Status ReadKaldiMatrix(Env* env, const string& ark_path, uint64 ark_offset, string* contents) {
        std::unique_ptr<RandomAccessFile> file_;
        std::unique_ptr<io::InputStreamInterface> buffered_inputstream_;
        enum { kBufferSize = 256 << 10 /* 256 kB */ };

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
        TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(3, &header_buffer));
        if (header_buffer == "CM ") {
            // format 1

            // Reading global_header
            string global_header;
            uint64 global_header_sz = 4 * 4;
            uint64 per_col_header_sz = 2 * 4;
            TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(global_header_sz, &global_header));
            int32 format = 1;
            float min_value, range;
            int32 num_rows, num_cols;
            memcpy(&min_value, global_header.data()+ 4 * 0, sizeof(float));
            memcpy(&range, global_header.data()    + 4 * 1, sizeof(float));
            memcpy(&num_rows, global_header.data() + 4 * 2, sizeof(int32));
            memcpy(&num_cols, global_header.data() + 4 * 3, sizeof(int32));

            // Calculate record size
            uint64 size = global_header_sz + num_cols * (per_col_header_sz + num_rows);
            uint64 remaining_size = size - global_header_sz;
            string data;
            TF_RETURN_IF_ERROR(buffered_inputstream_->ReadNBytes(remaining_size, &data));
            *contents = header_buffer + global_header + data;
        } else if (header_buffer == "DM ") {
            return Status(error::UNAVAILABLE, "Kaldi Matrix of double reading is not implemented yet.");
        } else if (header_buffer == "FM "){
            string row_and_col;
            buffered_inputstream_->ReadNBytes(1+4+1+4, &row_and_col);
            int32 row, col;
            memcpy(&row, row_and_col.data()+1, sizeof(int32));
            memcpy(&col, row_and_col.data()+6, sizeof(int32));
            string data;
            buffered_inputstream_->ReadNBytes(row * col * sizeof(float), &data);
            *contents = header_buffer + row_and_col + data;
        } else {
            return Status(error::UNAVAILABLE, "Unknown Kaldi Matrix: " + header_buffer);
        }
        return Status::OK();
    }


    class ReadKaldiMatrixOp : public OpKernel {
    public:
        using OpKernel::OpKernel;
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
                           ReadKaldiMatrix(context->env(), ark_path, ark_offset,
                                           &output->scalar<string>()()));
        }
    };
    REGISTER_KERNEL_BUILDER(Name("ReadKaldiMatrix").Device(DEVICE_CPU), ReadKaldiMatrixOp);

    REGISTER_OP("ReadKaldiMatrix")
            .Input("scpline: string")
            .Output("contents: string")
            .SetShapeFn(ScalarInputsAndOutputs)
            .Doc(R"doc(
Reads and outputs the contents of a record of the input kaldi ark filename.

scpline: scalar. /path/to/ark.file:12345
)doc");

    class DecodeKaldiMatrixOp : public OpKernel {
    public:
        explicit DecodeKaldiMatrixOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
        }

        void Compute(OpKernelContext* context) override {
            const auto& input = context->input(0);
            int64 str_size = -1;
            auto flat_in = input.flat<string>();
            OP_REQUIRES(context, flat_in.size() == 1,
                        errors::InvalidArgument(
                                "DecodeKaldiArk requires input string size = 1"
                        )
            );
            const string& in_str = flat_in(0);
            str_size = in_str.size();

            const char* in_data = reinterpret_cast<const char*>(flat_in(0).data());
            TensorShape out_shape;
            int32 num_elem = 0;
            if (in_data[0] == 'C' && in_data[1] == 'M') {
                float min_value    = *reinterpret_cast<const float*>(in_data + 3 + 4*0);
                float range        = *reinterpret_cast<const float*>(in_data + 3 + 4*1);
                int32 num_rows     = *reinterpret_cast<const int32*>(in_data + 3 + 4*2);
                int32 num_cols     = *reinterpret_cast<const int32*>(in_data + 3 + 4*3);
                out_shape.AddDim(num_rows);
                out_shape.AddDim(num_cols);
                num_elem = num_rows * num_cols;
            } else if (in_data[0] == 'F' && in_data[1] == 'M') {
                int32 num_rows = *reinterpret_cast<const int32*>(in_data + 3 + 1);
                int32 num_cols = *reinterpret_cast<const int32*>(in_data + 3 + 1 + 4 + 1);
                out_shape.AddDim(num_rows);
                out_shape.AddDim(num_cols);
                num_elem = num_rows * num_cols;
            }
            if (str_size == -1 || str_size == 0) {  // Empty input
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(context, context->allocate_output("output", out_shape,
                                                                 &output_tensor));
                return;
            }

            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(
                    context, context->allocate_output("output", out_shape, &output_tensor));
            auto out = output_tensor->flat<float>();

            float* out_data = out.data();
            if (in_data[0] == 'C' && in_data[1] == 'M') {
                GlobalHeader header;
                header.format       = 1;
                header.min_value    = *reinterpret_cast<const float*>(in_data + 3 + 4*0);
                header.range        = *reinterpret_cast<const float*>(in_data + 3 + 4*1);
                header.num_rows     = *reinterpret_cast<const int32*>(in_data + 3 + 4*2);
                header.num_cols     = *reinterpret_cast<const int32*>(in_data + 3 + 4*3);
                const PerColHeader *per_col_header = reinterpret_cast<const PerColHeader*>(in_data + 3 + 4*4);
                const uint8* in_data_bytes = reinterpret_cast<const uint8*>(per_col_header + header.num_cols);

                for (int32 i = 0; i < header.num_cols; i++, per_col_header++) {
                    float   p0 = Uint16ToFloat(header, per_col_header->percentile_0),
                            p25 = Uint16ToFloat(header, per_col_header->percentile_25),
                            p75 = Uint16ToFloat(header, per_col_header->percentile_75),
                            p100 = Uint16ToFloat(header, per_col_header->percentile_100);

                    for (int32 j = 0; j < header.num_rows; j ++, in_data_bytes ++) {
                        float f = CharToFloat(p0, p25, p75, p100, *in_data_bytes);
                        *(out_data + j * header.num_cols + i) = f;
                    }
                }
            } else if (in_data[0] == 'F' && in_data[1] == 'M') {
                memcpy(out_data, in_data + 3 + 10, num_elem * sizeof(float));
            }
        }

    private:
        DataType out_type_;
        struct PerColHeader {
            uint16 percentile_0;
            uint16 percentile_25;
            uint16 percentile_75;
            uint16 percentile_100;
        };
        struct GlobalHeader {
            int32 format;
            float min_value;
            float range;
            int32 num_rows;
            int32 num_cols;
        };
        float Uint16ToFloat(const GlobalHeader &global_header, uint16 value) {
            return global_header.min_value
                   + global_header.range * 1.52590218966964e-05F * value;
        }
        float CharToFloat(float p0, float p25, float p75, float p100,
                          uint8 value) {
            if (value <= 64) {
                return p0 + (p25 - p0) * value * (1/64.0f);
            } else if (value <= 192) {
                return p25 + (p75 - p25) * (value - 64) * (1/128.0f);
            } else {
                return p75 + (p100 - p75) * (value - 192) * (1/63.0f);
            }
        }
    };

    REGISTER_KERNEL_BUILDER(Name("DecodeKaldiMatrix").Device(DEVICE_CPU), DecodeKaldiMatrixOp);


    REGISTER_OP("DecodeKaldiMatrix")
            .Input("bytes: string")
            .Output("output: out_type")
            .Attr("out_type: {float}")
            .SetShapeFn(shape_inference::UnknownShape)
            .Doc(R"doc(
Reinterpret the bytes of a string as a kaldi matrix
)doc");
}  // namespace tensorflow
