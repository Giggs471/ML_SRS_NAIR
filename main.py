import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from logging import getLogger, INFO, Formatter, StreamHandler
import os

# Custom modules
from data_loader.loader import DataReader
from preprocessing.data_cleaner import DataProcessor
from preprocessing.feature_engineer import FeatureCreator
from preprocessing.vectorizer import DataTransformer
from modelling.classifiers import ModelBuilder
from evaluation.metrics import PerformanceAnalyzer
from visualization.plot_results import plot_metrics_bar

# Logger setup
main_logger = getLogger("PipelineRunner")
main_logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
main_logger.addHandler(handler)

# Configuration
DATA_FILE = "bots_vs_users.csv"
TARGET = "target"
SPLIT_RATIO = 0.2
SEED = 42
MODELS = ["logistic", "rf", "gb"]
PLOT_DIR = "plots"
SHOW_PLOTS = False  # Set to True to display plots interactively

def run_pipeline():
    """Execute the user vs bot classification pipeline."""
    main_logger.info("Initiating classification pipeline...")

    # Step 1: Load data
    main_logger.info("Loading dataset...")
    reader = DataReader(DATA_FILE)
    data = reader.read_csv_data()
    if data is None:
        main_logger.error("Failed to load dataset. Exiting.")
        return

    main_logger.info(f"Loaded columns: {data.columns.tolist()}")
    if TARGET not in data.columns:
        main_logger.error(f"Target column '{TARGET}' not found. Columns: {data.columns.tolist()}")
        return

    # Ensure target is numeric
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data.dropna(subset=[TARGET], inplace=True)
    data[TARGET] = data[TARGET].astype(int)
    main_logger.info(f"Target distribution:\n{data[TARGET].value_counts()}")

    # Step 2: Feature engineering
    main_logger.info("Performing feature engineering...")
    features = data.drop(TARGET, axis=1)
    labels = data[TARGET]
    feat_creator = FeatureCreator()
    features = feat_creator.generate_features(features)
    main_logger.info(f"Features after engineering: {features.columns.tolist()}")

    # Step 3: Split data
    main_logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=SPLIT_RATIO, random_state=SEED, stratify=labels
    )
    main_logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Step 4: Preprocessing
    main_logger.info("Preprocessing data...")
    processor = DataProcessor()
    num_cols, cat_cols = processor.detect_column_types(X_train, TARGET)
    processor.initialize_imputers(X_train, num_cols, cat_cols)

    # Process training data
    X_train_clean = processor.process_data(X_train, TARGET)
    y_train = y_train.loc[X_train_clean.index]  # Align y_train with X_train_clean

    # Process test data
    X_test_clean = processor.process_data(X_test, TARGET)
    y_test = y_test.loc[X_test_clean.index]  # Align y_test with X_test_clean

    main_logger.info(f"Numerical columns after preprocessing: {num_cols}")
    main_logger.info(f"Categorical columns after preprocessing: {cat_cols}")

    vectorizer = DataTransformer(num_columns=num_cols, cat_columns=cat_cols)
    vectorizer.prepare(X_train_clean)

    # Step 5: Train models
    main_logger.info("Training classification models...")
    builder = ModelBuilder(model_types=MODELS, seed=SEED)
    trained_models = builder.build_and_train(vectorizer.transform_pipeline, X_train_clean, y_train)

    if not trained_models:
        main_logger.error("No models trained. Exiting.")
        return

    # Step 6: Evaluate models
    main_logger.info("Evaluating models...")
    analyzer = PerformanceAnalyzer(trained_models)
    results = analyzer.assess_models(X_test_clean, y_test)
    analyzer.display_metrics()

    # Step 7: Visualize results
    main_logger.info("Generating visualizations...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    main_logger.info(f"Plots will be saved to: {os.path.abspath(PLOT_DIR)}")

    # ROC curves
    analyzer.visualize_roc(y_test, PLOT_DIR, show=SHOW_PLOTS)

    # Feature importances
    analyzer.visualize_importances(PLOT_DIR, max_features=25, show=SHOW_PLOTS)

    # Classification scatter plots
    main_logger.info(f"Available numerical columns for scatter plots: {num_cols}")
    if len(num_cols) >= 2:
        for model_name in trained_models.keys():
            analyzer.visualize_classification(X_test_clean, y_test, model_name,
                                             feat_x=num_cols[0], feat_y=num_cols[1], output_dir=PLOT_DIR, show=SHOW_PLOTS)
    else:
        main_logger.warning("Not enough numerical features for classification visualization. Need at least 2 numerical columns.")

    # Metrics bar plots
    for model_name, metrics in results.items():
        if "error" in metrics:
            continue
        plot_metrics_bar({k: v for k, v in metrics.items() if k in ["accuracy", "precision", "recall", "f1"]},
                         model_name, os.path.join(PLOT_DIR, f"metrics_{model_name}.png"), show=SHOW_PLOTS)

    # Step 8: Detailed reports
    main_logger.info("Generating detailed classification reports...")
    print("\n===== Classification Reports =====")
    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"\n{model_name}: Error - {metrics['error']}")
        else:
            print(f"\n{model_name}:")
            print(classification_report(y_test, metrics["predictions"], zero_division=0))

    # Step 9: Example prediction
    main_logger.info("Performing example predictions...")
    best_model = None
    best_f1 = -1
    for name, metrics in results.items():
        if "error" not in metrics and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = name

    if best_model:
        sample_data = X_test_clean.iloc[:2]
        predictions = trained_models[best_model].predict(sample_data)
        probabilities = trained_models[best_model].predict_proba(sample_data) if hasattr(trained_models[best_model], "predict_proba") else None
        main_logger.info(f"Best model ({best_model}) predictions on sample: {predictions}")
        if probabilities is not None:
            main_logger.info(f"Probabilities: {probabilities[:, 1]}")

    # Step 10: Summary
    main_logger.info("Summarizing results...")
    print("\n=== Summary ===")
    print("Classification pipeline completed.")
    if best_model:
        print(f"Best model: {best_model} with F1 score: {best_f1:.4f}")
        print(f"Sample predictions: {predictions}")
        if probabilities is not None:
            print(f"Sample probabilities for class 1: {[f'{p:.2f}' for p in probabilities[:, 1]]}")
    print(f"Check the '{PLOT_DIR}' directory for visualizations.")
    print("===============")

if __name__ == "__main__":
    run_pipeline()