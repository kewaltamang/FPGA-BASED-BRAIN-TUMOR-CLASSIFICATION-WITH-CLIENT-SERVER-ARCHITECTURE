MODEL_PATH=./pb_models2
FROZEN_GRAPH=${MODEL_PATH}/frozen_graph.pb
INPUT_NODES=conv2d_input
OUTPUT_NODES=dense_2/Softmax
OUTPUT_DIR=./quantize_result2
ITERATIONS=16

vai_q_tensorflow quantize \
    --input_frozen_graph ${FROZEN_GRAPH} \
    --input_nodes ${INPUT_NODES} \
    --input_shapes ?,256,256,1 \
    --output_nodes ${OUTPUT_NODES} \
    --input_fn graph_input_fn.input_fn \
    --method 1 \
    --weight_bit 4\
    --activation_bit 4 \
    --calib_iter ${ITERATIONS} \
    --simulate_dpu 1 \
    --output_dir ${OUTPUT_DIR} \
    --dump_float 0