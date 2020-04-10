export TRT_RELEASE=/tensorrt
cd build 
cmake  -DCUDNN_ROOT_DIR=/workspace/cuDNN  -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out   ..
make -j4
make install
cd ../bin 
./trtexec   --deploy=../samples/deploy.prototxt  --output=output