
export TRT_RELEASE=/data/library/TensorRT-5.1.5.0
export TRT_SOURCE=/home/sensetime/workspace/TensorRT
export CUDNN_DIR=/data/library/cudnn

nvidia-docker run -v $TRT_RELEASE:/tensorrt -v $TRT_SOURCE:/workspace/TensorRT \
-v $CUDNN_DIR:/workspace/cuDNN -it registry.sensetime.com/model_conversion/caffe2trt:parser 
