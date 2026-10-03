# Logistic Regression Baseline

A binary classification baseline for comparison with a feed-forward neural network.

## Run in Google Colab

1. Open `logistic_regression.ipynb` in Google Colab.
2. Run the code cell and upload `dl_practical.csv` when prompted.
3. Read the printed F1, precision, recall, accuracy, prediction errors, and confusion matrix.

The Python script contains the same code and is also intended for Colab because it uses `google.colab.files.upload()`.

## Dataset

Supply a CSV with 10 numeric feature columns followed by a binary label column containing 0 or 1. It may have a header row or no header. Missing values are not supported. The dataset is not included.

## Method

- Stratified 70% training / 30% testing split, random seed 42.
- StandardScaler fitted only on the training data.
- LogisticRegression with max_iter=1000 and random_state=42.
- Class 1 is the positive class for precision, recall, and F1.
- Confusion matrix layout: [[true negatives, false positives], [false negatives, true positives]].

Use exactly the same train/test examples and preprocessing policy for the neural network comparison. Metrics are calculated at runtime; this code does not guarantee the results from an earlier screenshot. The 80% check is informational and does not replace the assignment's neural network requirement.

## Files

- `logistic_regression.ipynb`: ready-to-run Colab notebook.
- `logistic_regression.py`: the same baseline as a Python script.
- `requirements.txt`: numerical and machine-learning dependencies (Colab supplies its upload interface).
