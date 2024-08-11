import argparse
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from models.cnn_model import create_cnn_model

# Define the argument parser
parser = argparse.ArgumentParser(description='Convert CNN model with h5 encoded weights to pb')
parser.add_argument("--weights", required=True, help="./75_model_weights")
args = parser.parse_args()

# Clear any previous models in memory
K.clear_session()

# Set learning phase to 0 (test mode)
tf.keras.backend.set_learning_phase(0)

# Define model input shape and number of classes
input_shape = (256, 256, 1)  # Example input shape, adjust as needed
n_classes = 4

# Start a new TensorFlow session
with tf.compat.v1.Session() as sess:
    # Create the model
    model = create_cnn_model(input_shape, n_classes)
    
    # Load the weights into the model
    model.load_weights(args.weights)
    
    # Define the output directory and filenames
    output_graph_dir = os.path.dirname(args.weights)
    output_graph_file = os.path.join(output_graph_dir, "model.pb")
    
    # Get the input and output names
    input_names = [out.op.name for out in model.inputs]
    output_names = [out.op.name for out in model.outputs]
    print(f"Input node: {input_names}")
    print(f"Output node: {output_names}")
    
    # Save the model
    saver = tf.compat.v1.train.Saver()
    graph_def = sess.graph.as_graph_def()
    save_path = saver.save(sess, os.path.join(output_graph_dir, "float_model.ckpt"))
    
    tf.io.write_graph(graph_def, output_graph_dir, "infer_graph.pb", as_text=False)
    
    # Output confirmation
    print(f"##### Model saved to {output_graph_file} #####")
    print(model.summary())
