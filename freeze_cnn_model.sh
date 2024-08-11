
MODEL_PATH=./pb_models2
OUTPUT_NAME=dense_2/Softmax

freeze_graph \
  --input_graph=${MODEL_PATH}/infer_graph.pb \
  --input_checkpoint=${MODEL_PATH}/float_model.ckpt \
  --output_graph=${MODEL_PATH}/frozen_graph.pb \
  --output_node_names=${OUTPUT_NAME} \
  --input_meta_graph=${MODEL_PATH}/float_model.ckpt.meta \
  --input_binary=true
