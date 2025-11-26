# IDS_25msccs057

Cybersecurity Intrusion Detection
Project Overview

This project focuses on building and evaluating various machine learning models to detect cybersecurity intrusions based on network session data. The goal is to identify malicious activities indicated by the attack_detected feature.
Dataset

The dataset cybersecurity_intrusion_data.csv contains various features related to network sessions, including network_packet_size, protocol_type, login_attempts, session_duration, encryption_used, ip_reputation_score, failed_logins, browser_type, unusual_time_access, and attack_detected (the target variable).
Dal data exploration (head, info, describe).

Dependencies

To run this notebook, you will need the following Python libraries:

   pandas
    numpy
    scikit-learn
    tensorflow (for Keras)
    keras-bert (if BERT specific features were planned/used, though not directly in the final model shown)
    keras-rectified-adam (if RAdam optimizer specific features were planned/used, though not directly in the final model shown)

You can install these using pip:

pip install pandas numpy scikit-learn tensorflow keras-bert keras-rectified-adam

Setup and Usage

   Clone the Repository (if applicable):

   git clone <your-repository-url>
    cd <your-repository-name>

   Upload the Dataset: Ensure the cybersecurity_intrusion_data.csv file is in the same directory as your Jupyter/Colab notebook, or update the path in the code.

   Run the Notebook: Open and run the Jupyter Notebook or Google Colab notebook cells sequentially. The notebook performs the following steps:
        Loads the dataset.
        Performs initial data exploration (head, info, describe).
        Drops irrelevant columns (session_id, encryption_used, browser_type, protocol_type).
        Splits the data into training and testing sets.
        Trains several classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Random Forest, and a simple Artificial Neural Network (ANN) using Keras.
        Evaluates each model and prints its accuracy, classification report, and confusion matrix.
        Saves the trained scikit-learn models as .pkl files.
Drops irrelevant columns (session_id, encryption_used, browser_type, protocol_type).
Splits the data into training and testing sets.
Trains several classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Random Forest, and a simple Artificial Neural Network (ANN) using Keras. Evaluates each model and prints its accuracy, classification report, and confusion matrix.
Saves the trained scikit-learn models as .pkl files.

Models and Performance

The following classification models were trained and evaluated:

   Logistic Regression:
        Accuracy: 0.737
        F1-score (weighted avg): 0.74
    Decision Tree:
        Accuracy: 0.785
        F1-score (weighted avg): 0.78
    Random Forest:
        Accuracy: 0.867
        F1-score (weighted avg): 0.86
    K-Nearest Neighbors (KNN):
        Accuracy: 0.516
        F1-score (weighted avg): 0.51
    Artificial Neural Network (ANN) - Keras:
        Accuracy: 0.724
        F1-score (weighted avg): 0.72

Random Forest achieved the highest accuracy among the trained models for this dataset.
Saved Models

The trained scikit-learn models are saved as pickle files in the root directory:

  logistic_regression.pkl
    decision_tree.pkl
    random_forest.pkl
    knn.pkl
    ann.pkl (Note: The Keras ANN model is generally saved using model.save(), not pickle.dump(). The code here pickled the MLPClassifier from scikit-learn which was named ann)
