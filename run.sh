export TRT_RELEASE=/tensorrt
export CPLUS_INCLUDE_PATH=/workspace/cuDNN/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/workspace/TensorRT/lib/:/tensorrt/lib:/workspace/cuDNN/lib64/:$LD_LIBRARY_PATH
cd build 
cmake -DCMAKE_BUILD_TYPE=Release  -DCUDNN_ROOT_DIR=/workspace/cuDNN  -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_BIN_DIR=`pwd`/out   ..
make -j4
make install
cd ../bin 
# ./trtexec_debug    --deploy=../samples/deploy.prototxt  --output=Split_191_0  --output=Split_191_1
./trtexec    --deploy=../samples/deploy.prototxt  --output=prob