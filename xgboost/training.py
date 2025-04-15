import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import csv
import sys
import time
import psutil
import json
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

class SystemMonitor:
    """Monitor system resources during training"""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu_percentages = []
        self.memory_usage = []
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.cpu_percentages = []
        self.memory_usage = []

    def update(self):
        """Record current system stats"""
        self.cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

    def stop(self):
        """Stop monitoring and return stats"""
        self.end_time = time.time()
        return {
            'training_time_seconds': self.end_time - self.start_time,
            'avg_cpu_percent': np.mean(self.cpu_percentages),
            'max_cpu_percent': max(self.cpu_percentages),
            'avg_memory_mb': np.mean(self.memory_usage),
            'max_memory_mb': max(self.memory_usage)
        }

def load_and_combine_data(file_paths):
    """
    Load and combine data from multiple CSV files,
    handling large field sizes.

    Args:
        file_paths (list): List of paths to CSV files containing malware data
    """
    # Increase field size limit to handle large API sequences
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    dfs = []
    required_columns = ['first_api', 'last_api', 'api_call_count',
                       'api_sequence', 'malware_type']

    for file_path in file_paths:
        try:
            # Use the 'python' engine to handle potential parsing issues
            df = pd.read_csv(file_path, engine='python')

            # Validate required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"File {file_path} is missing required columns: {missing_columns}")

            # Basic data cleaning
            df = df.dropna()
            df['api_call_count'] = df['api_call_count'].astype(int)
            df['api_sequence'] = df['api_sequence'].astype(str)

            # Add source file information
            df['source_file'] = os.path.basename(file_path)

            dfs.append(df)

            print(f"\nLoaded data from {file_path}")
            print(f"Samples: {len(df)}")
            print("Class distribution:")
            for malware_type, count in df['malware_type'].value_counts().items():
                print(f"{malware_type}: {count} samples")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Print combined dataset statistics
    print("\nCombined Dataset Statistics:")
    print("-" * 50)
    print(f"Total samples: {len(combined_df)}")
    print("\nOverall class distribution:")
    class_dist = combined_df['malware_type'].value_counts()
    for malware_type, count in class_dist.items():
        print(f"{malware_type}: {count} samples")

    # API calls statistics
    print("\nAPI calls statistics:")
    api_stats = combined_df['api_call_count'].describe()
    print(f"Min API calls: {api_stats['min']:.0f}")
    print(f"Max API calls: {api_stats['max']:.0f}")
    print(f"Mean API calls: {api_stats['mean']:.0f}")
    print(f"Median API calls: {api_stats['50%']:.0f}")

    return combined_df

def prepare_data(df, api_vocab_file, max_features=2000, use_smote=True):
    """
    Prepare the data using extended API vocabulary.
    """
    first_api_encoder = ExtendedAPIEncoder()
    last_api_encoder = ExtendedAPIEncoder()
    malware_type_encoder = LabelEncoder()

    print("Loading API vocabulary...")
    first_api_encoder.load_vocabulary(api_vocab_file)
    last_api_encoder.load_vocabulary(api_vocab_file)

    print("Encoding API calls...")
    df['first_api_encoded'] = first_api_encoder.fit_transform(df['first_api'])
    df['last_api_encoded'] = last_api_encoder.fit_transform(df['last_api'])
    df['malware_type_encoded'] = malware_type_encoder.fit_transform(df['malware_type'])

    df['api_call_count_norm'] = np.log1p(df['api_call_count'])

    print("Creating TF-IDF features...")
    with open(api_vocab_file, 'r') as f:
        vocabulary = {line.strip() for line in f if line.strip()}

    tfidf = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=1,
        vocabulary=vocabulary
    )
    api_sequence_features = tfidf.fit_transform(df['api_sequence'])

    numeric_features = np.column_stack((
        df['first_api_encoded'],
        df['last_api_encoded'],
        df['api_call_count_norm']
    ))
    numeric_features_sparse = csr_matrix(numeric_features)

    X = hstack([numeric_features_sparse, api_sequence_features])
    y = df['malware_type_encoded']

    if use_smote:
        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"Shape after SMOTE: {X.shape}")

    feature_names = (['First API', 'Last API', 'API Call Count'] +
                    [f'API_Seq_{i}' for i in range(api_sequence_features.shape[1])])

    return (X, y, malware_type_encoder, first_api_encoder, last_api_encoder,
            tfidf, feature_names)


def format_classification_report(y_true, y_pred, target_names):
    """Generate formatted classification report with 4 decimal precision"""
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    formatted_report = {"precision": {}, "recall": {}, "f1-score": {}, "support": {}}

    for class_name in target_names:
        metrics = report_dict[class_name]
        formatted_report["precision"][class_name] = f"{metrics['precision']:.4f}"
        formatted_report["recall"][class_name] = f"{metrics['recall']:.4f}"
        formatted_report["f1-score"][class_name] = f"{metrics['f1-score']:.4f}"
        formatted_report["support"][class_name] = int(metrics['support'])

    # Add weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        metrics = report_dict[avg_type]
        formatted_report["precision"][avg_type] = f"{metrics['precision']:.4f}"
        formatted_report["recall"][avg_type] = f"{metrics['recall']:.4f}"
        formatted_report["f1-score"][avg_type] = f"{metrics['f1-score']:.4f}"
        formatted_report["support"][avg_type] = int(metrics['support'])

    return formatted_report

def train_xgboost(X, y, num_class, system_monitor=None):
    """
    Enhanced XGBoost training with system monitoring and early stopping
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=num_class,
        tree_method='hist',
        eval_metric=['mlogloss', 'merror'],
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )

    cv_scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        fold_start_time = time.time()

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        eval_set = [(X_val, y_val)]

        xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )

        score = xgb_model.score(X_val, y_val)
        cv_scores.append(score)

        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} accuracy: {score:.4f} (Time: {fold_time:.2f}s)")

        if system_monitor:
            system_monitor.update()

    print(f"\nMean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final training on full dataset
    eval_set = [(X, y)]
    xgb_model.fit(X, y, eval_set=eval_set, verbose=100)

    return xgb_model

def evaluate_model(model, X, y, malware_type_encoder, feature_names, output_dir, system_monitor):
    """
    Enhanced evaluation with system metrics and precise formatting
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    class_names = malware_type_encoder.classes_

    # Generate formatted classification report
    formatted_report = format_classification_report(y_test, y_pred, class_names)

    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, 'model_metrics.json')
    system_stats = system_monitor.stop()

    metrics_data = {
        'classification_metrics': formatted_report,
        'system_stats': system_stats,
        'timestamp': datetime.now().isoformat()
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    # Print formatted results
    print("\nClassification Metrics (4 decimal precision):")
    print("-" * 80)
    for metric in ["precision", "recall", "f1-score"]:
        print(f"\n{metric.upper()}:")
        for class_name, value in formatted_report[metric].items():
            print(f"{class_name}: {value}")

    print("\nSystem Statistics:")
    print(f"Total training time: {system_stats['training_time_seconds']:.2f} seconds")
    print(f"Average CPU usage: {system_stats['avg_cpu_percent']:.1f}%")
    print(f"Peak CPU usage: {system_stats['max_cpu_percent']:.1f}%")
    print(f"Average memory usage: {system_stats['avg_memory_mb']:.1f} MB")
    print(f"Peak memory usage: {system_stats['max_memory_mb']:.1f} MB")

    # Create confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Feature importance plot
    n_top_features = 20
    importance_type = 'weight'
    importance_scores = model.get_booster().get_score(importance_type=importance_type)

    feature_importance_list = [(feature, importance_scores.get(feature, 0))
                              for feature in feature_names]

    feature_importance = pd.DataFrame(feature_importance_list, columns=['feature', 'importance'])
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance = feature_importance.head(n_top_features)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top {n_top_features} Most Important Features ({importance_type})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def save_artifacts(output_dir, model, tfidf, malware_type_encoder,
                  first_api_encoder, last_api_encoder, feature_names):
    """
    Save all model artifacts required for inference.
    """
    os.makedirs(output_dir, exist_ok=True)

    artifact_dict = {
        'malware_classifier.json': lambda x: x.save_model(os.path.join(output_dir, 'malware_classifier.json')),
        'tfidf_vectorizer.joblib': lambda x: joblib.dump(x, os.path.join(output_dir, 'tfidf_vectorizer.joblib')),
        'malware_type_encoder.joblib': lambda x: joblib.dump(x, os.path.join(output_dir, 'malware_type_encoder.joblib')),
        'first_api_encoder.joblib': lambda x: joblib.dump(x, os.path.join(output_dir, 'first_api_encoder.joblib')),
        'last_api_encoder.joblib': lambda x: joblib.dump(x, os.path.join(output_dir, 'last_api_encoder.joblib')),
        'feature_names.joblib': lambda x: joblib.dump(x, os.path.join(output_dir, 'feature_names.joblib'))
    }

    artifacts = [model, tfidf, malware_type_encoder, first_api_encoder, last_api_encoder, feature_names]

    for (filename, save_func), artifact in zip(artifact_dict.items(), artifacts):
        try:
            save_func(artifact)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")

def main(csv_paths, output_dir, api_vocab_file):
    """
    Enhanced main execution function with system monitoring
    """
    system_monitor = SystemMonitor()
    system_monitor.start()

    # Load and prepare data
    df = load_and_combine_data(csv_paths)
    (X, y, malware_type_encoder, first_api_encoder, last_api_encoder,
     tfidf, feature_names) = prepare_data(df, api_vocab_file, max_features=2000, use_smote=True)

    # Get number of classes for XGBoost
    num_classes = len(np.unique(y))

    # Train model with system monitoring
    model = train_xgboost(X, y, num_classes, system_monitor)

    # Evaluate model with system metrics
    evaluate_model(model, X, y, malware_type_encoder, feature_names, output_dir, system_monitor)

    # Save artifacts
    save_artifacts(
        output_dir,
        model,
        tfidf,
        malware_type_encoder,
        first_api_encoder,
        last_api_encoder,
        feature_names
    )

    print(f"\nTraining complete. All artifacts saved to {output_dir}")
    return model

if __name__ == "__main__":
    csv_paths = [
        "/csv/path/data.csv"
    ]
    output_dir = "output/directory/"
    api_vocab_file = "/api/calls/path/windowsapicalls.txt"
    model = main(csv_paths, output_dir, api_vocab_file)
