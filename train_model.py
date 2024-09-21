import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize

# Create output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # Probability estimates for all classes

    # Accuracy and Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save accuracy and classification report to a text file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))  # Save plot
    plt.close()  # Close the plot

    # ROC Curve for each class
    n_classes = y_prob.shape[1]
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (class {0} AUC = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))  # Save plot
    plt.close()  # Close the plot

    # Feature Importance
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X_test.shape[1]), feature_importances[indices], align='center')
    plt.xticks(range(X_test.shape[1]), np.array(X_test.columns)[indices], rotation=90)
    plt.xlim([-1, X_test.shape[1]])
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))  # Save plot
    plt.close()  # Close the plot



















