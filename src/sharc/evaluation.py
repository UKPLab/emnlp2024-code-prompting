from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def evaluate(list_y, list_y_pred):
    # calculate accuracy with scikit learn
    accuracy = accuracy_score(list_y, list_y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # calculate f1 score with scikit learn
    f1 = f1_score(list_y, list_y_pred, average="weighted")
    print(f"F1 Score: {f1:.2f}")
    # calculate confusion matrix with scikit learn

    # plot confusion matrix
    cm = confusion_matrix(list_y, list_y_pred)
    return {"accuracy": accuracy, "f1": f1, "confusion_matrix": cm.tolist()}

def save_confusion_matrix(cm, output_path):
    cm = np.array(cm)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    # classes are No, Unknown, Yes
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=["No", "Unknown", "Yes"],
        yticklabels=["No", "Unknown", "Yes"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    # save figure
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
