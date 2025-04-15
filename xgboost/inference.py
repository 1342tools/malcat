import pandas as pd
import numpy as np
from xgboost import XGBClassifier, Booster, DMatrix
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import time
import psutil
from datetime import datetime

class ExtendedAPIEncoder:
    """
    Custom encoder for API calls that handles unseen values using a predefined vocabulary.
    """
    def __init__(self, unknown_value=-1):
        self.label_encoder = LabelEncoder()
        self.unknown_value = unknown_value
        self.vocabulary = set()

    def load_vocabulary(self, vocab_file):
        """Load API vocabulary from a text file"""
        with open(vocab_file, 'r') as f:
            api_calls = {line.strip() for line in f if line.strip()}
        self.vocabulary.update(api_calls)

    def add_to_vocabulary(self, api_calls):
        """Add additional API calls to vocabulary"""
        self.vocabulary.update(api_calls)

    def fit(self, api_calls):
        """Fit the encoder using both the vocabulary and training data"""
        all_apis = list(self.vocabulary.union(set(api_calls)))
        self.label_encoder.fit(all_apis)
        return self

    def transform(self, api_calls):
        """Transform API calls, handling unseen values gracefully"""
        api_calls_clean = np.array(api_calls).copy()
        mask = ~np.isin(api_calls_clean, self.label_encoder.classes_)
        if mask.any():
            unseen_apis = set(api_calls_clean[mask])
            print(f"Warning: Found {len(unseen_apis)} unseen API calls not in vocabulary.")
            api_calls_clean[mask] = self.label_encoder.classes_[0]

        return self.label_encoder.transform(api_calls_clean)

    def fit_transform(self, api_calls):
        """Fit and transform in one step"""
        self.fit(api_calls)
        return self.transform(api_calls)

    def inverse_transform(self, encoded_values):
        """Convert encoded values back to API calls"""
        return self.label_encoder.inverse_transform(encoded_values)

    def classes_(self):
        """Return the classes (API calls) known to the encoder"""
        return self.label_encoder.classes_

def create_run_directory(base_output_dir):
    """
    Create a timestamped directory for the current run.

    Args:
        base_output_dir (str): Base directory for all runs

    Returns:
        str: Path to the newly created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_output_dir, f'run_{timestamp}')

    # Create directory structure
    subdirs = ['individual_results', 'confusion_matrices', 'combined_results', 'metrics']
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir

def get_system_metrics():
    """
    Collect system metrics during model execution.
    """
    process = psutil.Process()
    return {
        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(),
        'threads': process.num_threads(),
    }

def load_model_artifacts(model_dir):
    """
    Load all saved model artifacts required for inference.
    """
    try:
        # Load XGBoost model using lower-level API to avoid version issues
        booster = Booster()
        booster.load_model(os.path.join(model_dir, 'malware_classifier.json'))

        # Create XGBClassifier wrapper
        model = XGBClassifier()
        model._Booster = booster
        model.n_classes_ = len(booster.get_dump())  # Set number of classes

        # Load other artifacts
        tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        malware_type_encoder = joblib.load(os.path.join(model_dir, 'malware_type_encoder.joblib'))
        first_api_encoder = joblib.load(os.path.join(model_dir, 'first_api_encoder.joblib'))
        last_api_encoder = joblib.load(os.path.join(model_dir, 'last_api_encoder.joblib'))

        # Set objective for XGBClassifier
        model.objective = 'multi:softprob'

        return model, tfidf, malware_type_encoder, first_api_encoder, last_api_encoder

    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        raise

def prepare_batch_samples(df, tfidf, first_api_encoder, last_api_encoder):
    """
    Prepare multiple samples for prediction.
    """
    try:
        # Encode first and last API calls
        first_api_encoded = first_api_encoder.transform(df['first_api'])
        last_api_encoded = last_api_encoder.transform(df['last_api'])

        # Normalize API call count
        api_call_count_norm = np.log1p(df['api_call_count'])

        # Create TF-IDF features for API sequence
        api_sequence_features = tfidf.transform(df['api_sequence'])

        # Combine features
        numeric_features = np.column_stack((
            first_api_encoded,
            last_api_encoded,
            api_call_count_norm
        ))
        numeric_features_sparse = csr_matrix(numeric_features)

        # Create final feature matrix
        X = hstack([numeric_features_sparse, api_sequence_features])

        return X
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def plot_confusion_matrix(true_labels, predicted_labels, class_names, output_path_prefix):
    """
    Create and save confusion matrix visualization.

    Args:
        output_path_prefix (str): Base path for saving confusion matrices (without extension)
    """
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Raw counts matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix (Raw Counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_counts.png")
        plt.close()

        # Normalized matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.4%', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_normalized.png")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {str(e)}")
        raise

def process_single_file(model_dir, file_path, run_dir):
    """
    Process a single CSV file and save results.
    """
    start_time = time.time()
    initial_metrics = get_system_metrics()
    file_name = os.path.basename(file_path)

    try:
        print(f"\nProcessing {file_name}...")

        # Load data
        df = pd.read_csv(file_path)
        required_columns = ['first_api', 'last_api', 'api_call_count', 'api_sequence', 'malware_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Load model and artifacts
        model, tfidf, malware_type_encoder, first_api_encoder, last_api_encoder = \
            load_model_artifacts(model_dir)

        # Prepare features
        X = prepare_batch_samples(df, tfidf, first_api_encoder, last_api_encoder)

        # Make predictions
        dmatrix = DMatrix(X)
        pred_proba = model.get_booster().predict(dmatrix)
        pred_classes = pred_proba.argmax(axis=1)
        predicted_labels = malware_type_encoder.inverse_transform(pred_classes)

        # Prepare results DataFrame
        results_df = df.copy()
        results_df['predicted_malware_type'] = predicted_labels
        results_df['confidence'] = [max(probs) for probs in pred_proba]

        # Add class probabilities
        for i, class_name in enumerate(malware_type_encoder.classes_):
            results_df[f'prob_{class_name}'] = pred_proba[:, i]

        # Calculate metrics
        accuracy = (predicted_labels == df['malware_type']).mean()
        file_metrics = {
            'filename': file_name,
            'processing_time': time.time() - start_time,
            'sample_count': len(df),
            'accuracy': round(accuracy, 4),
            'avg_confidence': results_df['confidence'].mean(),
            'low_confidence_count': sum(results_df['confidence'] < 0.5)
        }

        # Create confusion matrix
        output_path_prefix = os.path.join(run_dir, 'confusion_matrices', f'confusion_matrix_{file_name[:-4]}')
        plot_confusion_matrix(
            df['malware_type'],
            predicted_labels,
            malware_type_encoder.classes_,
            output_path_prefix
        )

        # Save classification report
        report = classification_report(df['malware_type'], predicted_labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(run_dir, 'metrics', f'classification_report_{file_name}'), index=True)

        # Add system metrics
        final_metrics = get_system_metrics()
        file_metrics.update({
            'peak_memory_mb': max(initial_metrics['memory_usage_mb'], final_metrics['memory_usage_mb']),
            'peak_cpu_percent': max(initial_metrics['cpu_percent'], final_metrics['cpu_percent']),
            'peak_threads': max(initial_metrics['threads'], final_metrics['threads'])
        })

        # Save individual results
        output_path = os.path.join(run_dir, 'individual_results', f'predictions_{file_name}')
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        return results_df, file_metrics

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None, None

def process_multiple_files(model_dir, csv_paths, base_output_dir):
    """
    Process multiple CSV files and generate combined analysis.
    """
    # Create directory for this run
    run_dir = create_run_directory(base_output_dir)
    print(f"Created new run directory: {run_dir}")

    # Process each file individually
    all_metrics = []
    all_dfs = []

    for file_path in csv_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        df, metrics = process_single_file(model_dir, file_path, run_dir)

        if df is not None and metrics is not None:
            all_dfs.append(df)
            all_metrics.append(metrics)

    if not all_dfs:
        raise ValueError("No files were processed successfully")

    # Process combined dataset
    print("\nProcessing combined dataset...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate combined metrics
    combined_metrics = {
        'filename': 'combined_dataset',
        'sample_count': len(combined_df),
        'accuracy': round((combined_df['malware_type'] == combined_df['predicted_malware_type']).mean(), 4),
        'avg_confidence': combined_df['confidence'].mean(),
        'low_confidence_count': sum(combined_df['confidence'] < 0.5)
    }

    # Create combined confusion matrix
    plot_confusion_matrix(
        combined_df['malware_type'],
        combined_df['predicted_malware_type'],
        combined_df['predicted_malware_type'].unique(),
        os.path.join(run_dir, 'combined_results', 'confusion_matrix_combined')
    )

    # Save combined results
    combined_output = os.path.join(run_dir, 'combined_results', 'combined_predictions.csv')
    combined_df.to_csv(combined_output, index=False)

    # Create summary report
    summary = pd.DataFrame(all_metrics + [combined_metrics])
    summary.to_csv(os.path.join(run_dir, 'metrics', 'processing_metrics.csv'), index=False)

    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    for metrics in all_metrics:
        print(f"\nFile: {metrics['filename']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Processing Time: {metrics['processing_time']:.2f} seconds")
        print(f"Peak Memory Usage: {metrics['peak_memory_mb']:.2f} MB")
        print(f"Peak CPU Usage: {metrics['peak_cpu_percent']:.2f}%")

    print("\nCombined Dataset Results:")
    print(f"Total Samples: {combined_metrics['sample_count']}")
    print(f"Overall Accuracy: {combined_metrics['accuracy']:.4f}")
    print(f"Average Confidence: {combined_metrics['avg_confidence']:.4f}")

    print(f"\nAll results saved in: {run_dir}")

if __name__ == "__main__":
    # Specific CSV paths
    csv_paths = [
        "/csv/path/dataset.csv"
    ]

    model_dir = '/model/directory/'
    base_output_dir = '/output/directory/'

    process_multiple_files(model_dir, csv_paths, base_output_dir)
