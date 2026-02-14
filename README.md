Convolutional Neural Networks for Image Classification
This repository demonstrates the design, implementation, and evaluation of Convolutional Neural Network (CNN) architectures using industry-standard deep learning frameworks. The project explores both custom-built CNN models and transfer learning approaches to analyze performance, generalization, and overfitting behavior on image classification tasks.
🚀 Key Objectives
Design and train custom CNN architectures using Keras and PyTorch
Implement Global Average Pooling (GAP) to reduce model complexity and overfitting
Apply transfer learning using pre-trained architectures such as ResNet and VGG
Compare from-scratch CNNs vs pre-trained models
Evaluate model performance using standard classification metrics
🏗️ Model Architectures
🔹 Custom CNN
Convolution + ReLU blocks
Max Pooling layers
Global Average Pooling instead of dense fully connected layers
Dropout for regularization
🔹 Transfer Learning
Pre-trained models:
ResNet (residual learning)
VGG (deep feature extraction)
Frozen base layers with custom classification head
Fine-tuning for improved domain adaptation
📊 Model Evaluation
Models are evaluated using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Performance comparison highlights the trade-offs between training time, generalization, and accuracy.
📁 Tech Stack
Frameworks: Keras (TensorFlow), PyTorch
Libraries: NumPy, Pandas, Matplotlib, Scikit-learn
Concepts: CNNs, Transfer Learning, Global Average Pooling, Regularization
📌 Key Takeaways
Global Average Pooling significantly reduces overfitting in custom CNNs
Transfer learning achieves higher accuracy with less training data
Pre-trained models generalize better but require careful fine-tuning
Custom CNNs offer flexibility and faster experimentation
