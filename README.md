## Dog vs Cat Classifier using CNN

This project is a deep learning model that classifies images as either dogs or cats using a Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras, and it leverages image data from the Kaggle "Dogs vs Cats" dataset. The project includes data preprocessing, model creation, training, and evaluation.

### Key Features:
- **Data Preprocessing**: Images are resized and normalized for model input.
- **CNN Architecture**: The model consists of multiple convolutional layers, max-pooling layers, batch normalization, dropout layers, and dense layers.
- **Training**: The model is trained for 10 epochs with data augmentation and normalization.
- **Evaluation**: The model's performance is evaluated using accuracy and loss metrics, visualized through plots.

### Tools & Libraries:
- TensorFlow
- Keras
- Matplotlib (for visualization)

### Dataset:
- The dataset used is from Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats).

### How to Use:
1. Clone the repository.
2. Install the required libraries (`tensorflow`, `keras`, `matplotlib`).
3. Download the dataset from Kaggle and place it in the appropriate directory.
4. Run the Jupyter notebook to preprocess the data, train the model, and evaluate its performance.

### Results:
- The model achieves a validation accuracy of approximately 82.9% after 10 epochs.

### Future Improvements:
- Experiment with different CNN architectures.
- Increase the number of epochs for better accuracy.
- Use transfer learning with pre-trained models (e.g., ResNet, VGG).

Feel free to contribute or suggest improvements! 🐾
