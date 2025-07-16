#!/usr/bin/env python3
"""
Sensor Anomaly Detection using LSTM Autoencoder

This script converts the Jupyter notebook "Sensor Anomaly Detection.ipynb" 
into a well-structured Python script for anomaly detection in sensor data
using LSTM autoencoder neural networks.

Author: Converted from Jupyter notebook
"""

import argparse
import logging
import os
import sys
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(10)
tf.random.set_seed(10)

# Configure TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure plotting
sns.set(color_codes=True)
plt.style.use('default')


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV data and perform initial preprocessing.
    
    Args:
        data_path: Path to the CSV file containing sensor data
        
    Returns:
        DataFrame with preprocessed sensor data
        
    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If the data format is invalid
    """
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Load the data
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.index = data['timestamp']
        
        # Extract sensor columns (excluding timestamp and NaN columns)
        sensor_columns = ['YT.11FI_02044.PV', 'YT.11PIC_02044.PV', 'YT.11TI_02044.PV']
        available_columns = [col for col in sensor_columns if col in data.columns]
        
        if not available_columns:
            raise ValueError("No valid sensor columns found in the data")
        
        # Select only sensor data columns
        processed_data = data[available_columns].copy()
        
        # Remove any rows with all NaN values
        processed_data = processed_data.dropna(how='all')
        
        logger.info(f"Data preprocessing completed. Final shape: {processed_data.shape}")
        logger.info(f"Available sensor columns: {available_columns}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def split_data(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Input DataFrame with sensor data
        train_ratio: Ratio of data to use for training (default: 0.8)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    logger.info(f"Splitting data with train ratio: {train_ratio}")
    
    total_len = len(data)
    split_idx = int(total_len * train_ratio)
    
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    
    logger.info(f"Training dataset shape: {train_data.shape}")
    logger.info(f"Test dataset shape: {test_data.shape}")
    
    return train_data, test_data


def visualize_data(data: pd.DataFrame, title: str = "Sensor Data", 
                  save_path: Optional[str] = None) -> None:
    """
    Create time series visualization of sensor data.
    
    Args:
        data: DataFrame with sensor data
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    logger.info(f"Creating visualization: {title}")
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    
    # Normalize and plot each sensor column
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    labels = ['flow', 'pressure', 'temperature', 'sensor4', 'sensor5']
    
    for i, col in enumerate(data.columns[:min(len(colors), len(data.columns))]):
        if col in data.columns:
            normalized_data = (data[col] - data[col].mean()) / data[col].std()
            ax.plot(normalized_data, label=labels[i], color=colors[i], linewidth=1)
    
    plt.legend(loc='lower left')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Values')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    
    plt.show()


def visualize_frequency_data(data: pd.DataFrame, title: str = "Frequency Data",
                           save_path: Optional[str] = None) -> None:
    """
    Create frequency domain visualization using FFT.
    
    Args:
        data: DataFrame with sensor data
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    logger.info(f"Creating frequency domain visualization: {title}")
    
    # Apply FFT
    data_normalized = (data - data.mean()) / data.std()
    data_fft = np.fft.fft(data_normalized)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    labels = ['flow', 'pressure', 'temperature', 'sensor4', 'sensor5']
    
    for i, col in enumerate(data.columns[:min(len(colors), len(data.columns))]):
        if i < len(data_fft[0]):
            if i % 2 == 0:
                ax.plot(data_fft[:, i].real, label=labels[i], color=colors[i], linewidth=1)
            else:
                ax.plot(data_fft[:, i].imag, label=labels[i], color=colors[i], linewidth=1)
    
    plt.legend(loc='lower left')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Frequency plot saved to: {save_path}")
    
    plt.show()


def normalize_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply StandardScaler normalization to the data.
    
    Args:
        train_data: Training data DataFrame
        test_data: Test data DataFrame
        
    Returns:
        Tuple of (X_train_normalized, X_test_normalized, scaler)
    """
    logger.info("Normalizing data using StandardScaler")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)
    X_test = scaler.transform(test_data)
    
    # Reshape for LSTM input [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    logger.info(f"Training data shape after normalization: {X_train.shape}")
    logger.info(f"Test data shape after normalization: {X_test.shape}")
    
    return X_train, X_test, scaler


def create_autoencoder_model(input_shape: Tuple[int, int]) -> Model:
    """
    Define the LSTM autoencoder architecture.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        
    Returns:
        Compiled Keras model
    """
    logger.info(f"Creating autoencoder model with input shape: {input_shape}")
    
    inputs = Input(shape=input_shape)
    
    # Encoder
    L1 = LSTM(32, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    
    # Decoder
    L3 = RepeatVector(input_shape[0])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(input_shape[1]))(L5)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    
    logger.info("Model created and compiled successfully")
    logger.info(f"Model summary:\n{model.summary()}")
    
    return model


def train_model(model: Model, X_train: np.ndarray, epochs: int = 15, 
                batch_size: int = 128, validation_split: float = 0.1) -> Dict[str, Any]:
    """
    Train the autoencoder model.
    
    Args:
        model: Keras model to train
        X_train: Training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        Training history dictionary
    """
    logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
    
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    ).history
    
    logger.info("Model training completed")
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model Loss', fontsize=16)
    ax.set_ylabel('Loss (MAE)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()
    
    return history


def calculate_threshold(model: Model, X_train: np.ndarray, train_data: pd.DataFrame, 
                       k: float = 3.0) -> float:
    """
    Determine anomaly detection threshold based on training data reconstruction loss.
    
    Args:
        model: Trained autoencoder model
        X_train: Training data (normalized and reshaped)
        train_data: Original training data DataFrame
        k: Multiplier for standard deviation in threshold calculation
        
    Returns:
        Calculated threshold value
    """
    logger.info(f"Calculating anomaly threshold with k={k}")
    
    # Get predictions on training data
    X_pred_train = model.predict(X_train)
    X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
    X_pred_train = pd.DataFrame(X_pred_train, columns=train_data.columns, index=train_data.index)
    
    # Calculate reconstruction loss
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2])
    loss_mae = np.mean(np.abs(X_pred_train - X_train_reshaped), axis=1)
    
    # Calculate threshold
    threshold = np.mean(loss_mae) + k * np.std(loss_mae)
    
    logger.info(f"Calculated threshold: {threshold}")
    
    # Plot loss distribution
    plt.figure(figsize=(16, 9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.histplot(loss_mae, bins=20, kde=True, color='blue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlim([-0.1, 0.3])
    plt.legend()
    plt.show()
    
    return threshold


def detect_anomalies(model: Model, X_test: np.ndarray, test_data: pd.DataFrame, 
                    threshold: float) -> pd.DataFrame:
    """
    Perform anomaly detection on test data.
    
    Args:
        model: Trained autoencoder model
        X_test: Test data (normalized and reshaped)
        test_data: Original test data DataFrame
        threshold: Anomaly detection threshold
        
    Returns:
        DataFrame with anomaly detection results
    """
    logger.info("Performing anomaly detection on test data")
    
    # Get predictions on test data
    X_pred_test = model.predict(X_test)
    X_pred_test = X_pred_test.reshape(X_pred_test.shape[0], X_pred_test.shape[2])
    X_pred_test = pd.DataFrame(X_pred_test, columns=test_data.columns, index=test_data.index)
    
    # Calculate reconstruction loss
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[2])
    loss_mae = np.mean(np.abs(X_pred_test - X_test_reshaped), axis=1)
    
    # Create results DataFrame
    results = pd.DataFrame(index=test_data.index)
    results['Loss_mae'] = loss_mae
    results['Threshold'] = threshold
    results['Anomaly'] = results['Loss_mae'] > threshold
    
    anomaly_count = results['Anomaly'].sum()
    anomaly_percentage = (anomaly_count / len(results)) * 100
    
    logger.info(f"Detected {anomaly_count} anomalies ({anomaly_percentage:.2f}% of test data)")
    
    return results


def visualize_results(train_results: pd.DataFrame, test_results: pd.DataFrame,
                     save_path: Optional[str] = None) -> None:
    """
    Plot loss distribution and anomaly detection results.
    
    Args:
        train_results: Training data results with loss and anomaly flags
        test_results: Test data results with loss and anomaly flags
        save_path: Optional path to save the plot
    """
    logger.info("Creating anomaly detection results visualization")
    
    # Combine results
    combined_results = pd.concat([train_results, test_results])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    combined_results[['Loss_mae', 'Threshold']].plot(
        logy=True, 
        figsize=(16, 9), 
        ylim=[1e-2, 1e2], 
        color=['blue', 'red'],
        ax=ax
    )
    
    # Highlight anomalies
    anomaly_points = combined_results[combined_results['Anomaly']]
    if len(anomaly_points) > 0:
        ax.scatter(anomaly_points.index, anomaly_points['Loss_mae'], 
                  color='red', alpha=0.7, s=10, label='Anomalies')
    
    ax.set_title('Anomaly Detection Results', fontsize=16)
    ax.set_ylabel('Loss (MAE)')
    ax.set_xlabel('Time')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Results plot saved to: {save_path}")
    
    plt.show()


def save_model(model: Model, model_path: str, scaler: StandardScaler, 
               scaler_path: Optional[str] = None) -> None:
    """
    Save the trained model and scaler.
    
    Args:
        model: Trained Keras model
        model_path: Path to save the model
        scaler: Fitted StandardScaler
        scaler_path: Optional path to save the scaler
    """
    logger.info(f"Saving model to: {model_path}")
    
    # Save model
    model.save(model_path)
    logger.info("Model saved successfully")
    
    # Save scaler if path provided
    if scaler_path:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")


def main():
    """Main function that orchestrates the entire workflow."""
    parser = argparse.ArgumentParser(description='Sensor Anomaly Detection using LSTM Autoencoder')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the CSV file containing sensor data')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--threshold-k', type=float, default=3.0,
                        help='Multiplier for threshold calculation (default: 3.0)')
    parser.add_argument('--model-path', type=str, default='sensor_anomaly_model.h5',
                        help='Path to save the trained model (default: sensor_anomaly_model.h5)')
    parser.add_argument('--scaler-path', type=str, default='sensor_scaler.pkl',
                        help='Path to save the scaler (default: sensor_scaler.pkl)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save plots and results (default: output)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Load and preprocess data
        logger.info("=== Step 1: Loading and preprocessing data ===")
        data = load_and_preprocess_data(args.data_path)
        
        # Step 2: Split data
        logger.info("=== Step 2: Splitting data ===")
        train_data, test_data = split_data(data, args.train_ratio)
        
        # Step 3: Visualize data
        if not args.no_plots:
            logger.info("=== Step 3: Visualizing data ===")
            visualize_data(data, "Complete Sensor Data", 
                          os.path.join(args.output_dir, "sensor_data.png"))
            visualize_frequency_data(train_data, "Training Frequency Data",
                                   os.path.join(args.output_dir, "train_frequency.png"))
            visualize_frequency_data(test_data, "Test Frequency Data",
                                   os.path.join(args.output_dir, "test_frequency.png"))
        
        # Step 4: Normalize data
        logger.info("=== Step 4: Normalizing data ===")
        X_train, X_test, scaler = normalize_data(train_data, test_data)
        
        # Step 5: Create model
        logger.info("=== Step 5: Creating autoencoder model ===")
        model = create_autoencoder_model((X_train.shape[1], X_train.shape[2]))
        
        # Step 6: Train model
        logger.info("=== Step 6: Training model ===")
        history = train_model(model, X_train, args.epochs, args.batch_size)
        
        # Step 7: Calculate threshold
        logger.info("=== Step 7: Calculating anomaly threshold ===")
        threshold = calculate_threshold(model, X_train, train_data, args.threshold_k)
        
        # Step 8: Detect anomalies
        logger.info("=== Step 8: Detecting anomalies ===")
        train_results = pd.DataFrame(index=train_data.index)
        X_pred_train = model.predict(X_train)
        X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[2])
        train_results['Loss_mae'] = np.mean(np.abs(X_pred_train - X_train_reshaped), axis=1)
        train_results['Threshold'] = threshold
        train_results['Anomaly'] = train_results['Loss_mae'] > threshold
        
        test_results = detect_anomalies(model, X_test, test_data, threshold)
        
        # Step 9: Visualize results
        if not args.no_plots:
            logger.info("=== Step 9: Visualizing results ===")
            visualize_results(train_results, test_results,
                            os.path.join(args.output_dir, "anomaly_results.png"))
        
        # Step 10: Save model
        logger.info("=== Step 10: Saving model ===")
        save_model(model, args.model_path, scaler, args.scaler_path)
        
        # Save results to CSV
        results_path = os.path.join(args.output_dir, "anomaly_results.csv")
        combined_results = pd.concat([train_results, test_results])
        combined_results.to_csv(results_path)
        logger.info(f"Results saved to: {results_path}")
        
        logger.info("=== Anomaly detection pipeline completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()