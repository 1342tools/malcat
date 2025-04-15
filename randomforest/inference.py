import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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

def get_system_metrics():
    """
    Collect system metrics during model execution.

    Returns:
        dict: Dictionary containing system metrics
    """
    process = psutil.Process()
    return {
        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
        'cpu_percent': process.cpu_percent(),
        'threads': process.num_threads(),
    }

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
    os.makedirs(run_dir, exist_ok=True)

    # Create subdirectories for organization
    os.makedirs(os.path.join(run_dir, 'individual_results'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'combined_results'), exist_ok=True)

    return run_dir

def load_model_artifacts(model_dir):
    """
    Load all saved model artifacts from the specified directory.

    Args:
        model_dir (str): Directory containing the saved model artifacts

    Returns:
        tuple: (model, tfidf, malware_encoder, first_api_encoder, last_api_encoder, feature_names)
    """
    try:
        model = joblib.load(os.path.join(model_dir, 'malware_classifier.joblib'))
        tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        malware_encoder = joblib.load(os.path.join(model_dir, 'malware_type_encoder.joblib'))
        first_api_encoder = joblib.load(os.path.join(model_dir, 'first_api_encoder.joblib'))
        last_api_encoder = joblib.load(os.path.join(model_dir, 'last_api_encoder.joblib'))
        feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))

        return model, tfidf, malware_encoder, first_api_encoder, last_api_encoder, feature_names
    except Exception as e:
        raise Exception(f"Error loading model artifacts: {str(e)}")

def prepare_inference_data(df, first_api_encoder, last_api_encoder, tfidf):
    """
    Prepare new data for inference using extended API handling.
    """
    # Handle API calls using extended encoders
    first_api_encoded = first_api_encoder.transform(df['first_api'])
    last_api_encoded = last_api_encoder.transform(df['last_api'])

    # Normalize api_call_count
    api_call_count_norm = np.log1p(df['api_call_count'])

    # Transform API sequences using TF-IDF
    # The vectorizer will now handle unseen tokens using the predefined vocabulary
    api_sequence_features = tfidf.transform(df['api_sequence'])

    # Combine features
    numeric_features = np.column_stack((
        first_api_encoded,
        last_api_encoded,
        api_call_count_norm
    ))
    numeric_features_sparse = csr_matrix(numeric_features)

    return hstack([numeric_features_sparse, api_sequence_features])

def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None):
    """
    Create and save a confusion matrix visualization.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (array-like): List of class names
        output_path (str, optional): Path to save the confusion matrix plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure and axes
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(cm_percent, annot=True, fmt='.1f',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='YlOrRd')

    plt.title('Confusion Matrix (%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Rotate axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Confusion matrix plot saved to {output_path}")
    else:
        plt.show()

    plt.close()

def process_single_file(model_dir, file_path, run_dir):
    """
    Process a single CSV file and save results.

    Args:
        model_dir (str): Directory containing model artifacts
        file_path (str): Path to input CSV file
        run_dir (str): Directory for current run results

    Returns:
        tuple: (DataFrame with predictions, dict with metrics)
    """
    start_time = time.time()
    initial_metrics = get_system_metrics()
    file_name = os.path.basename(file_path)

    try:
        print(f"\nProcessing {file_name}...")

        # Load model and make predictions
        model, tfidf, malware_encoder, first_api_encoder, last_api_encoder, feature_names = load_model_artifacts(model_dir)
        df = pd.read_csv(file_path)

        # Prepare features and make predictions
        X = prepare_inference_data(df, first_api_encoder, last_api_encoder, tfidf)
        predictions = model.predict(X)
        prediction_probs = model.predict_proba(X)

        # Add predictions to DataFrame
        df['predicted_malware_type'] = malware_encoder.inverse_transform(predictions)
        for i, class_name in enumerate(malware_encoder.classes_):
            df[f'prob_{class_name}'] = prediction_probs[:, i]
        df['confidence'] = prediction_probs.max(axis=1)

        # Calculate metrics
        file_metrics = {
            'filename': file_name,
            'processing_time': time.time() - start_time,
            'sample_count': len(df),
            'avg_confidence': df['confidence'].mean(),
            'low_confidence_count': sum(df['confidence'] < 0.5)
        }

        if 'malware_type' in df.columns:
            accuracy = np.mean(df['malware_type'] == df['predicted_malware_type'])
            file_metrics['accuracy'] = round(accuracy, 4)

            # Create confusion matrix
            plot_confusion_matrix(
                df['malware_type'],
                df['predicted_malware_type'],
                malware_encoder.classes_,
                output_path=os.path.join(run_dir, 'confusion_matrices', f'confusion_matrix_{file_name}.png')
            )

        # Add system metrics
        final_metrics = get_system_metrics()
        file_metrics.update({
            'peak_memory_mb': max(initial_metrics['memory_usage_mb'], final_metrics['memory_usage_mb']),
            'peak_cpu_percent': max(initial_metrics['cpu_percent'], final_metrics['cpu_percent']),
            'peak_threads': max(initial_metrics['threads'], final_metrics['threads'])
        })

        # Save individual results
        output_path = os.path.join(run_dir, 'individual_results', f'predictions_{file_name}')
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        return df, file_metrics

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None, None

def process_multiple_files(model_dir, csv_paths, base_output_dir):
    """
    Process multiple CSV files and generate combined analysis.

    Args:
        model_dir (str): Directory containing model artifacts
        csv_paths (list): List of paths to CSV files
        base_output_dir (str): Base directory for all runs
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
    combined_metrics = {
        'filename': 'combined_dataset',
        'sample_count': len(combined_df),
        'avg_confidence': combined_df['confidence'].mean(),
        'low_confidence_count': sum(combined_df['confidence'] < 0.5)
    }

    if 'malware_type' in combined_df.columns:
        accuracy = np.mean(combined_df['malware_type'] == combined_df['predicted_malware_type'])
        combined_metrics['accuracy'] = round(accuracy, 4)

        # Create combined confusion matrix
        plot_confusion_matrix(
            combined_df['malware_type'],
            combined_df['predicted_malware_type'],
            combined_df['predicted_malware_type'].unique(),
            output_path=os.path.join(run_dir, 'combined_results', 'confusion_matrix_combined.png')
        )

    # Save combined results
    combined_output = os.path.join(run_dir, 'combined_results', 'combined_predictions.csv')
    combined_df.to_csv(combined_output, index=False)

    # Create summary report
    summary = pd.DataFrame(all_metrics + [combined_metrics])
    summary.to_csv(os.path.join(run_dir, 'processing_metrics.csv'), index=False)

    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    for metrics in all_metrics:
        print(f"\nFile: {metrics['filename']}")
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Processing Time: {metrics['processing_time']:.2f} seconds")
        print(f"Peak Memory Usage: {metrics['peak_memory_mb']:.2f} MB")
        print(f"Peak CPU Usage: {metrics['peak_cpu_percent']:.2f}%")

    print("\nCombined Dataset Results:")
    print(f"Total Samples: {combined_metrics['sample_count']}")
    if 'accuracy' in combined_metrics:
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

