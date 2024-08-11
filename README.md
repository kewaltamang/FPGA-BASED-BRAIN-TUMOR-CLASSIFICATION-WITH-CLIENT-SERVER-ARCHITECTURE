# FPGA-BASED-BRAIN-TUMOR-CLASSIFICATION-WITH-CLIENT-SERVER-ARCHITECTURE

This project features an advanced brain tumor classification system leveraging FPGA technology(Kria KR260 board) and a client-server architecture(Streamlit). The system is designed to perform real-time classification of brain tumors into four categories: Glioma Tumor, Pituitary Tumor, Meningioma Tumor, and No Tumor. By combining the computational power of FPGAs with a distributed client-server model, the solution provides rapid and accurate analysis of medical images, enhancing the diagnostic process in clinical environments.

KEY FEATURES:

FPGA Acceleration: Utilizes FPGA technology to accelerate the classification process, delivering high-speed inference with low latency.

Client-Server Architecture: Implements a distributed architecture where the server handles data processing and FPGA-based clients perform inference tasks.

Four-Class Classification: Accurately classifies brain tumors into Glioma, Pituitary Tumor, Meningioma, and No Tumor categories.

Real-Time Analysis: Provides fast, real-time classification, crucial for time-sensitive diagnostic applications.

Scalable and Efficient: The architecture is designed for scalability, allowing for the integration of multiple clients to handle larger workloads.


SYSTEM ARCHITECTURE:

Client: The FPGA acts as a client, performing the computationally intensive task of tumor classification. It receives input images from the server, processes them, and returns the classification results.

Server: The server manages the overall system, handling image pre-processing, distributing tasks to FPGA clients, and aggregating results. It also provides an interface for users to submit images and retrieve classification results.

USAGE
To deploy the system, follow the instructions in the Installation and Usage sections. The client-server architecture can be deployed in various healthcare environments, offering an efficient solution for brain tumor classification.

FUTURE ENHANCEMENT:
Integration of additional FPGA clients for increased processing power and redundancy.
Expansion of the model to include more tumor types and improve classification accuracy.
Optimization of the client-server communication protocol for faster data exchange and lower latency.
