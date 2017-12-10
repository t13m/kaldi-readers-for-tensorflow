cmake_minimum_required(VERSION 3.8)
project(kaldi_reader_standalone)

set(CMAKE_CXX_STANDARD 11)
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.1)
    # c++ regex is used in the code, so the version of gcc must be greater than 4.9
    message(FATAL_ERROR "VERSION OF GCC MUST BE GREATER THAN 5.1")
endif()

set(PYTHONBIN "python" CACHE PATH "Path of python with tensorflow installed")

execute_process(
        COMMAND ${PYTHONBIN} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE DEFAULT_TF_INC
        ERROR_VARIABLE ERROR_TF_INC
        RESULT_VARIABLE RESULT_TF_INC
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${PYTHONBIN} -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        OUTPUT_VARIABLE DEFAULT_TF_LIB
        ERROR_VARIABLE ERROR_TF_LIB
        RESULT_VARIABLE RESULT_TF_LIB
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
#message("TF_INC is set: ${DEFAULT_TF_INC}")
#message("TF_LIB is set: ${DEFAULT_TF_LIB}")
#set(TF_INC "${DEFAULT_TF_INC}" CACHE PATH "Path of tensorflow including files")
#set(TF_LIB "${DEFAULT_TF_LIB}" CACHE PATH "Path of tensorflow linking libraries")

set(TF_INC "${DEFAULT_TF_INC}")
set(TF_LIB "${DEFAULT_TF_LIB}")

if ("${TF_INC}" STREQUAL "" OR "${TF_LIB}" STREQUAL "")
    message(FATAL_ERROR "TF_INC and TF_LIB not set. Please set both variable manually, or set correct PYTHONBIN var.")
endif()


message("TF_INC is set: ${TF_INC}")
message("TF_LIB is set: ${TF_LIB}")

set(SOURCE_FILES
        kaldi-matrix.cc
        kaldi-matrix-direct.cc
        kaldi-ali.cc
        shape-funcs.cc
        )

add_library(kaldi_readers SHARED)
target_compile_definitions(kaldi_readers
        PUBLIC
        -D_GLIBCXX_USE_CXX11_ABI=0
        )
# -fPIC
set_property(TARGET kaldi_readers PROPERTY POSITION_INDEPENDENT_CODE ON)
target_sources(kaldi_readers
        PRIVATE
        ${SOURCE_FILES}
        )

target_include_directories(kaldi_readers
        PRIVATE
        ${TF_INC}
        ${TF_INC}/external/nsync/public
        )
target_link_libraries(kaldi_readers
        PRIVATE
        ${TF_LIB}/libtensorflow_framework.so
        )

