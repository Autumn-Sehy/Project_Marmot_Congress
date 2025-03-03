import csv
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, Perceptron
from data_processing import load_data_parallel

data_dir = "Data"


def write_predictions_to_csv(model_name, mc_predictions):
    """
    Write model predictions to a CSV file

    Args:
        model_name (str): Name of the model used for predictions
        mc_predictions (dict): Dictionary containing predictions per MC

    Returns:
        str: Path to the output CSV file
    """
    csv_rows = []
    for mc, counts in mc_predictions.items():
        label = max(counts, key=counts.get)
        if label == 'total':
            # Skip 'total' when determining the label
            label = max(['democrat', 'republican', 'freedom caucus'],
                        key=lambda k: counts[k])

        csv_row = {'mc': mc, 'num_files': counts['total'], 'democrat': int(label == 'democrat'),
                   'repub': int(label in ['republican', 'freedom caucus']),
                   'fc': int(label == 'freedom caucus'), 'label': label,
                   'percent_dem': f"{(counts['democrat'] / counts['total']) * 100:.2f}",
                   'percent_repub': f"{(counts['republican'] / counts['total']) * 100:.2f}",
                   'percent_fc': f"{(counts['freedom caucus'] / counts['total']) * 100:.2f}"}

        csv_rows.append(csv_row)

    # Write to CSV
    output_file = f"MC_classifier_{model_name.lower().replace(' ', '_')}_unified.csv"
    fieldnames = [
        'mc', 'num_files', 'democrat', 'republican',
        'freedom_caucus', 'label', 'perc_democrat',
        'perc_republican', 'perc_freedom_caucus'
    ]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return output_file


def tune_and_evaluate(base_model, model_name, param_grid, X_train_vect, y_train,
                      X_dev_vect, y_dev, X_test_vect, y_test, mcs_test):
    """
    Tune hyperparameters using grid search (see imports), evaluate on dev set,
    then test & write results to CSV.

    Args:
        base_model: The base model to tune
        model_name (str): Name of the model
        param_grid (dict): Parameter grid for GridSearchCV
        X_train_vect, y_train: Training data
        X_dev_vect, y_dev: Development data for validation
        X_test_vect, y_test: Test data for final eval
        mcs_test: MC IDs for test set

    Returns:
        str: Path to the output CSV file
    """
    print(f"\nTuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='f1_macro',
        verbose=1
    )

    # Fit on training data
    grid_search.fit(X_train_vect, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate on dev set
    print(f"\n{model_name} Development Set Performance:")
    dev_predictions = best_model.predict(X_dev_vect)
    print(classification_report(y_dev, dev_predictions))

    # Final evaluation on test set
    print(f"\n{model_name} Test Set Performance:")
    test_predictions = best_model.predict(X_test_vect)
    print(classification_report(y_test, test_predictions))

    # Aggregate speaker-level predictions
    mc_predictions = defaultdict(lambda: {
        'democrat': 0,
        'republican': 0,
        'freedom caucus': 0,
        'total': 0
    })

    for mc, pred in zip(mcs_test, test_predictions):
        mc_predictions[mc][pred] += 1
        mc_predictions[mc]['total'] += 1

    # Write predictions to CSV
    return write_predictions_to_csv(model_name, mc_predictions)


def run(multinomial_nb=True, logistic_regression=True, perceptron=True):
    """
    Pipeline:
    1. Load data 😎
    2. Train/dev/test split 🫥
    3. Vectorize 💃
    4. Tune hyperparameters on dev set 🕺
    5. Evaluate on test set && save predictions 🧐
    """
    print("Loading speeches!🤓 ")
    texts, labels, speakers = load_data_parallel(data_dir)
    print(f"~Loaded {len(texts)} speeches~")

    # Create Train, Dev, and Test splits
    X_train, X_temp, y_train, y_temp, speakers_train, speakers_temp = train_test_split(
        texts, labels, speakers, test_size=0.4, random_state=42, stratify=labels
    )
    X_dev, X_test, y_dev, y_test, speakers_dev, speakers_test = train_test_split(
        X_temp, y_temp, speakers_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

    # Vectorize
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_dev_vect = vectorizer.transform(X_dev)
    X_test_vect = vectorizer.transform(X_test)
    if multinomial_nb:
        nb_param_grid = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
            'fit_prior': [True, False]
        }
        tune_and_evaluate(
            MultinomialNB(), 'Naive_Bayes', nb_param_grid,
            X_train_vect, y_train, X_dev_vect, y_dev,
            X_test_vect, y_test, speakers_test
        )

    if logistic_regression:
        lr_param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        }
        tune_and_evaluate(
            LogisticRegression(), 'Logistic_Regression', lr_param_grid,
            X_train_vect, y_train, X_dev_vect, y_dev,
            X_test_vect, y_test, speakers_test
        )

    if perceptron:
        perceptron_param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'penalty': [None, 'l1', 'l2', 'elasticnet'],
            'max_iter': [500, 1000]
        }
        tune_and_evaluate(
            Perceptron(), 'Perceptron', perceptron_param_grid,
            X_train_vect, y_train, X_dev_vect, y_dev,
            X_test_vect, y_test, speakers_test
        )
