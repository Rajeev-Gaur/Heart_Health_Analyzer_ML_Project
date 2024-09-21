import pandas as pd
from preprocessing_data import preprocess_data
from train_model import train_random_forest, evaluate_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)  # Adjust pos_label as needed
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def main(file_path):
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Train the model
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)

if __name__ == "__main__":
    file_path = "Data/heart_disease_uci.csv"
    main(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)


