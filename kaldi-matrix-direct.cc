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

    class ReadAndDecodeKaldiMatrixOp : public OpKernel {
    public:

        using OpKernel::OpKernel;

        explicit ReadAndDecodeKaldiMatrixOp(OpKernelConstruction* context): OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("left_padding", &left_padding_));
            OP_REQUIRES_OK(context, context->GetAttr("right_padding", &right_padding_));
        }
        void Compute(OpKernelContext* context) override {

            const Tensor* input;
            OP_REQUIRES_OK(context, context->input("scpline", &input));
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                        errors::InvalidArgument(
                                "Input filename tensor must be scalar, but had shape: ",
                                input->shape().DebugString()));

            const std::regex id_pat("^(\\S+):(\\d+)");
            std::smatch m;
            string half_scp_line = input->scalar<string>()();
            bool matched = std::regex_search(half_scp_line, m, id_pat);
            OP_REQUIRES(context, matched, Status(error::INVALID_ARGUMENT, "Script line is " + half_scp_line));
            string ark_path = m[1];
            string ark_offset_str = m[2];
            uint64 ark_offset = std::stoull(ark_offset_str);

            std::unique_ptr<RandomAccessFile> file;
            OP_REQUIRES_OK(context, context->env()->NewRandomAccessFile(ark_path, &file));
            uint64 rel_offset = 0;
            StringPiece data_holder;
            char data_header[10];
            OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, 2, &data_holder, data_header));
            rel_offset += 2;
            bool is_binary = (data_header[0] == '\0' && data_header[1] == 'B');
            OP_REQUIRES(context, is_binary, Status(error::INVALID_ARGUMENT,
                                                   "We only support binary format ark."));

            OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, 3, &data_holder, data_header));
            rel_offset += 3;

            TensorShape out_shape;

            if (data_holder == "FM ") {
                int8 row_nbyte;
                int32 row;
                int8 col_nbyte;
                int32 col;
                OP_REQUIRES_OK(context,
                               file->Read(ark_offset + rel_offset, 1, &data_holder,
                                          reinterpret_cast<char*>(&row_nbyte)));
                rel_offset += 1;
                OP_REQUIRES_OK(context,
                               file->Read(ark_offset + rel_offset, 4, &data_holder,
                                          reinterpret_cast<char*>(&row)));
                rel_offset += 4;
                OP_REQUIRES_OK(context,
                               file->Read(ark_offset + rel_offset, 1, &data_holder,
                                          reinterpret_cast<char*>(&col_nbyte)));
                rel_offset += 1;
                OP_REQUIRES_OK(context,
                               file->Read(ark_offset + rel_offset, 4, &data_holder,
                                          reinterpret_cast<char*>(&col)));
                rel_offset += 4;

                out_shape.AddDim(left_padding_ + row + right_padding_);
                out_shape.AddDim(col);
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(
                        context, context->allocate_output("output", out_shape, &output_tensor));
                auto out = output_tensor->flat<float>();

                float* out_data = out.data();
                OP_REQUIRES_OK(context,
                               file->Read(ark_offset + rel_offset, row * col * sizeof(float), &data_holder,
                        reinterpret_cast<char*>(out_data + left_padding_ * col)));

                for (int64 i = 0; i < left_padding_; i ++) {
                    for (int j = 0; j < col; j ++) {
                        *(out_data + i * col + j) = *(out_data + left_padding_ * col + j);
                    }
                }
                for (int64 i = left_padding_ + row; i < left_padding_ + row + right_padding_; i ++) {
                    for (int j = 0; j < col; j ++) {
                        *(out_data + i * col + j) = *(out_data + (left_padding_ + row - 1) * col + j);
                    }
                }

            } else if (data_holder == "CM ") {
                GlobalHeader h;
                h.format = 1;
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, sizeof(h) - 4, &data_holder,
                        reinterpret_cast<char*>(&h) + 4));
                rel_offset += (sizeof(h) - 4);
                out_shape.AddDim(left_padding_ + h.num_rows + right_padding_);
                out_shape.AddDim(h.num_cols);
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(
                        context, context->allocate_output("output", out_shape, &output_tensor));
                auto out = output_tensor->flat<float>();

                uint64 remaining_size = h.num_cols * (h.num_rows + sizeof(PerColHeader));
                string compressed_buffer;
                compressed_buffer.resize(remaining_size);
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, remaining_size, &data_holder,
                                                   &compressed_buffer[0]));
                rel_offset += remaining_size;

                float* out_data = out.data();
                const char* in_data = compressed_buffer.data();

                const PerColHeader *per_col_header = reinterpret_cast<const PerColHeader*>(in_data);
                const uint8 *in_data_bytes = reinterpret_cast<const uint8*>(per_col_header + h.num_cols);
                for (int64 i = 0; i < h.num_cols; i++, per_col_header++) {
                    float   p0 = Uint16ToFloat(h, per_col_header->percentile_0),
                            p25 = Uint16ToFloat(h, per_col_header->percentile_25),
                            p75 = Uint16ToFloat(h, per_col_header->percentile_75),
                            p100 = Uint16ToFloat(h, per_col_header->percentile_100);

                    for (int64 j = left_padding_; j < left_padding_ + h.num_rows; j ++, in_data_bytes ++) {
                        float f = CharToFloat(p0, p25, p75, p100, *in_data_bytes);
                        *(out_data + j * h.num_cols + i) = f;
                    }
                }

                for (int64 i = 0; i < left_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + left_padding_ * h.num_cols + j);
                    }
                }
                for (int64 i = left_padding_ + h.num_rows; i < left_padding_ + h.num_rows + right_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + (left_padding_ + h.num_rows - 1) * h.num_cols + j);
                    }
                }
            } else if (data_holder == "CM2") {
                rel_offset ++;
                GlobalHeader h;
                h.format = 2;
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, sizeof(h) - 4, &data_holder,
                                                   reinterpret_cast<char*>(&h) + 4));
                rel_offset += (sizeof(h) - 4);
                out_shape.AddDim(left_padding_ + h.num_rows + right_padding_);
                out_shape.AddDim(h.num_cols);
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(
                        context, context->allocate_output("output", out_shape, &output_tensor));
                auto out = output_tensor->flat<float>();

                uint64 size = DataSize(h);
                uint64 remaining_size = size - sizeof(GlobalHeader);
                string compressed_buffer;
                compressed_buffer.resize(remaining_size);
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, remaining_size, &data_holder,
                                                   &compressed_buffer[0]));
                rel_offset += remaining_size;

                float* out_data = out.data();
                const char* in_data = compressed_buffer.data();

                const uint16 *in_data_uint16 = reinterpret_cast<const uint16*>(in_data);
                float min_value = h.min_value;
                float increment = h.range * (1.0 / 65535.0);
                for (int64 i = left_padding_; i < left_padding_ + h.num_rows; i++) {
                    for (int64 j = 0; j < h.num_cols; j++) {
                        *(out_data + i * h.num_cols + j) = min_value + in_data_uint16[j] * increment;
                    }
                    in_data_uint16 += h.num_cols;
                }
                for (int64 i = 0; i < left_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + left_padding_ * h.num_cols + j);
                    }
                }
                for (int64 i = left_padding_ + h.num_rows; i < left_padding_ + h.num_rows + right_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + (left_padding_ + h.num_rows - 1) * h.num_cols + j);
                    }
                }
            } else if (data_holder == "CM3") {
                rel_offset ++;
                GlobalHeader h;
                h.format = 3;
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, sizeof(h) - 4, &data_holder,
                                                   reinterpret_cast<char*>(&h) + 4));
                rel_offset += (sizeof(h) - 4);
                out_shape.AddDim(left_padding_ + h.num_rows + right_padding_);
                out_shape.AddDim(h.num_cols);
                Tensor* output_tensor = nullptr;
                OP_REQUIRES_OK(
                        context, context->allocate_output("output", out_shape, &output_tensor));
                auto out = output_tensor->flat<float>();

                uint64 size = DataSize(h);
                uint64 remaining_size = size - sizeof(GlobalHeader);
                string compressed_buffer;
                compressed_buffer.resize(remaining_size);
                OP_REQUIRES_OK(context, file->Read(ark_offset + rel_offset, remaining_size, &data_holder,
                                                   &compressed_buffer[0]));
                rel_offset += remaining_size;

                float* out_data = out.data();
                const char* in_data = compressed_buffer.data();

                float min_value = h.min_value, increment = h.range * (1.0 / 255.0);
                const uint8 *in_data_bytes = reinterpret_cast<const uint8*>(in_data);
                for (int64 i = left_padding_; i < left_padding_ + h.num_rows; i++) {
                    for (int64 j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = h.min_value + in_data_bytes[j] * increment;
                    }
                    in_data_bytes += h.num_cols;
                }
                for (int64 i = 0; i < left_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + left_padding_ * h.num_cols + j);
                    }
                }
                for (int64 i = left_padding_ + h.num_rows; i < left_padding_ + h.num_rows + right_padding_; i ++) {
                    for (int j = 0; j < h.num_cols; j ++) {
                        *(out_data + i * h.num_cols + j) = *(out_data + (left_padding_ + h.num_rows - 1) * h.num_cols + j);
                    }
                }
            } else {
                OP_REQUIRES_OK(context, Status(error::UNAVAILABLE,
                                               "Unknown Kaldi Matrix:" + data_holder.ToString() +
                                               " When reading \"" + half_scp_line + "\"" +
                                               " Ark: " + ark_path +
                                               " OFFSET: " + std::to_string(ark_offset) ));
            }
        }
    private:
        int64 left_padding_, right_padding_;
        enum DataFormat {
            kOneByteWithColHeaders = 1,
            kTwoByte = 2,
            kOneByte = 3
        };
        struct GlobalHeader {
            int32 format;     // Represents the enum DataFormat.
            float min_value;  // min_value and range represent the ranges of the integer
            // data in the kTwoByte and kOneByte formats, and the
            // range of the PerColHeader uint16's in the
            // kOneByteWithColheaders format.
            float range;
            int32 num_rows;
            int32 num_cols;
        };
        struct PerColHeader {
            uint16 percentile_0;
            uint16 percentile_25;
            uint16 percentile_75;
            uint16 percentile_100;
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
        uint64 DataSize(const GlobalHeader& header) {
            DataFormat format = static_cast<DataFormat>(header.format);
            if (format == kOneByteWithColHeaders) {
                return sizeof(GlobalHeader) +
                       header.num_cols * (sizeof(PerColHeader) + header.num_rows);
            } else if (format == kTwoByte) {
                return sizeof(GlobalHeader) +
                       2 * header.num_rows * header.num_cols;
            } else {
                return sizeof(GlobalHeader) +
                       header.num_rows * header.num_cols;
            }
        }
    };
    REGISTER_KERNEL_BUILDER(Name("ReadAndDecodeKaldiMatrix").Device(DEVICE_CPU), ReadAndDecodeKaldiMatrixOp);


    REGISTER_OP("ReadAndDecodeKaldiMatrix")
            .Input("scpline: string")
            .Attr("left_padding: int")
            .Attr("right_padding: int")
            .Output("output: float32")
            .SetShapeFn(shape_inference::UnknownShape)
            .Doc(R"doc(
Reinterpret the bytes of a string as a kaldi matrix
)doc");
}
