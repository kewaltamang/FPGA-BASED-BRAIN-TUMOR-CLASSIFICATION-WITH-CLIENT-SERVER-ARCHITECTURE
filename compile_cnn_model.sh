# Author: Mark Harvey

#ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KriaKR260/arch.json # Path to the architecture file for the specific DPU
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json 
QUANT=quantize_result2  # Path to the directory containing the quantized model
COMPILE_CNN=compile_results2  # Output directory for the compiled model
NET_NAME=cnn_model  # Name of the network
LOG=logs  # Directory for log files
COMP_LOG_CNN=compile_cnn.log  # Log file name for the compilation process

# Function to compile the model
compile() {
  vai_c_tensorflow \
    --frozen_pb ${QUANT}/quantize_eval_model.pb \
    --arch ${ARCH} \
    --output_dir ${COMPILE_CNN} \
    --net_name ${NET_NAME}
}

echo "-----------------------------------------"
echo "COMPILE cnn STARTED.."
echo "-----------------------------------------"

# Remove the output directory if it exists and create a new one
rm -rf ${COMPILE_CNN}
mkdir -p ${COMPILE_CNN}

# Create log directory if it doesn't exist
mkdir -p ${LOG}

# Run the compile function and log the output
if compile 2>&1 | tee ${LOG}/${COMP_LOG_CNN}; then
  echo "-----------------------------------------"
  echo "COMPILE cnn COMPLETED"
  echo "-----------------------------------------"
else
  echo "-----------------------------------------"
  echo "COMPILE cnn FAILED"
  echo "-----------------------------------------"
  exit 1
fi
