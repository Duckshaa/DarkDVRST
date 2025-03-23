import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_generator):
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(test_generator.labels, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
