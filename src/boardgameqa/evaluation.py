from sklearn.metrics import accuracy_score, confusion_matrix


class Answer:
    YES = 1
    NO = -1
    UNKNOWN = 0


def evaluate(dataset, list_predictions):
    """
    Evaluates the accuracy of predictions using scikit-learn.

    Args:
        dataset (list): A list of dictionaries representing the dataset.
            Each dictionary should have a "label" key indicating the true label.
        list_predictions (list): A list of predicted labels.

    Returns:
        dict: A dictionary containing the accuracy score and confusion matrix.

    """
    list_labels = []
    for x in dataset:
        if x["label"] == "proved":
            list_labels.append(Answer.YES)
        elif x["label"] == "disproved":
            list_labels.append(Answer.NO)
        else:
            list_labels.append(Answer.UNKNOWN)

    # compute confusion matrix
    cm = confusion_matrix(list_labels, list_predictions).tolist()
    return {
        "accuracy": accuracy_score(list_labels, list_predictions) * 100,
        "confusion_matrix": cm,
    }
