#!/bin/bash

TEST_NAME=resnet32_cifar100
ONNX_NAME=${TEST_NAME}_pre.onnx
WRK_DIR=`pwd`

[ $# -ne 1 ] && echo "[WARN] Usage: $0 <cfg_name>. For example: $0 test" && exit 0
[ ! -f driver/fhe_cmplr ] && echo "[ERROR] Run $o under the build directory" && exit 1
[ ! -f ../fhe-cmplr/rtlib/ant/dataset/$TEST_NAME.cxx ] && echo "[ERROR] failed to locate $TEST_NAME main file" && exit 1
[ ! -f ../model/$ONNX_NAME ] && echo "[ERROR] failed to locate ../model/$ONNX_NAME" && exit 1

ID=$1
Q0=60
SF=56

RELU_VR_DEF=3
RELU_VR="/relu/Relu=5"    #  1
RELU_VR="$RELU_VR;/layer1/layer1.0/relu_1/Relu=6"    #  3
RELU_VR="$RELU_VR;/layer1/layer1.1/relu_1/Relu=7"    #  5
RELU_VR="$RELU_VR;/layer1/layer1.2/relu_1/Relu=8"    #  7
RELU_VR="$RELU_VR;/layer1/layer1.3/relu_1/Relu=10"    #  9
RELU_VR="$RELU_VR;/layer1/layer1.4/relu/Relu=4"    #  10
RELU_VR="$RELU_VR;/layer1/layer1.4/relu_1/Relu=7"    #  11
RELU_VR="$RELU_VR;/layer2/layer2.0/relu/Relu=4"    #  12
RELU_VR="$RELU_VR;/layer2/layer2.0/relu_1/Relu=6"    #  13
RELU_VR="$RELU_VR;/layer2/layer2.1/relu_1/Relu=8"    #  15
RELU_VR="$RELU_VR;/layer2/layer2.2/relu/Relu=4"    #  16
RELU_VR="$RELU_VR;/layer2/layer2.2/relu_1/Relu=8"    #  17
RELU_VR="$RELU_VR;/layer2/layer2.3/relu_1/Relu=9"    #  19
RELU_VR="$RELU_VR;/layer2/layer2.4/relu_1/Relu=11"    #  21
RELU_VR="$RELU_VR;/layer3/layer3.0/relu/Relu=4"    #  22
RELU_VR="$RELU_VR;/layer3/layer3.0/relu_1/Relu=8"    #  23
RELU_VR="$RELU_VR;/layer3/layer3.1/relu_1/Relu=9"    #  25
RELU_VR="$RELU_VR;/layer3/layer3.2/relu/Relu=4"    #  26
RELU_VR="$RELU_VR;/layer3/layer3.2/relu_1/Relu=11"    #  27
RELU_VR="$RELU_VR;/layer3/layer3.3/relu/Relu=4"    #  28
RELU_VR="$RELU_VR;/layer3/layer3.3/relu_1/Relu=26"    #  29
RELU_VR="$RELU_VR;/layer3/layer3.4/relu/Relu=5"    #  30
RELU_VR="$RELU_VR;/layer3/layer3.4/relu_1/Relu=46"    #  31

VEC_OPTS="-VEC:rtt:conv_fast"
SIHE_OPTS="-SIHE:relu_vr_def=$RELU_VR_DEF:relu_vr=$RELU_VR"
CKKS_OPTS="-CKKS:sk_hw=192:q0=$Q0:sf=$SF"
P2C_OPTS="-P2C:df=$WRK_DIR/$ONNX_NAME.$ID.msg:fp"

./driver/fhe_cmplr ../model/$ONNX_NAME $VEC_OPTS $SIHE_OPTS $CKKS_OPTS $P2C_OPTS
[ $? -ne 0 ] && echo "[ERROR] fhe_cmplr failed. exit" && exit 1

cp -v ${ONNX_NAME}.c ../fhe-cmplr/rtlib/ant/dataset/${ONNX_NAME}.inc
make $TEST_NAME
[ $? -ne 0 ] && echo "[ERROR] build $TEST_NAME failed. exit" && exit 1
[ ! -f ./dataset/$TEST_NAME ] && echo "[ERROR] failed to locate ./dataset/$TEST_NAME" && exit 1
cp ./dataset/$TEST_NAME ./dataset/$TEST_NAME.$ID

echo "[INFO] build $TEST_NAME succeed."
exit 0
