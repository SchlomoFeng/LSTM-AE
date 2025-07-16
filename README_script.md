# Sensor Anomaly Detection using LSTM Autoencoder

This project provides a well-structured Python script for sensor anomaly detection using LSTM autoencoder neural networks. The script was converted from the original Jupyter notebook "Sensor Anomaly Detection.ipynb" to provide better modularity, reusability, and maintainability.

## Features

- **Modular Design**: All functionality is organized into well-defined functions
- **Command Line Interface**: Full CLI support with configurable parameters
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Visualization**: Automatic generation of plots for data analysis and results
- **Model Persistence**: Automatic saving of trained models and scalers
- **Error Handling**: Robust error handling and validation

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python sensor_anomaly_detection.py --data-path "your_data.csv"
```

### Advanced Usage

```bash
python sensor_anomaly_detection.py \
    --data-path "气化一期S4_imputed.csv" \
    --train-ratio 0.8 \
    --epochs 15 \
    --batch-size 128 \
    --threshold-k 3.0 \
    --model-path "my_model.h5" \
    --scaler-path "my_scaler.pkl" \
    --output-dir "results"
```

### Command Line Arguments

- `--data-path`: Path to the CSV file containing sensor data (required)
- `--train-ratio`: Ratio of data to use for training (default: 0.8)
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size for training (default: 128)
- `--threshold-k`: Multiplier for threshold calculation (default: 3.0)
- `--model-path`: Path to save the trained model (default: sensor_anomaly_model.h5)
- `--scaler-path`: Path to save the scaler (default: sensor_scaler.pkl)
- `--output-dir`: Directory to save plots and results (default: output)
- `--no-plots`: Skip generating plots

## Script Functions

The script is organized into the following main functions:

### Data Processing Functions
- `load_and_preprocess_data()`: Load CSV data and perform initial preprocessing
- `split_data()`: Split data into train/test sets
- `normalize_data()`: Apply StandardScaler normalization

### Visualization Functions
- `visualize_data()`: Create time series plots of sensor data
- `visualize_frequency_data()`: Create frequency domain plots using FFT
- `visualize_results()`: Plot loss distribution and anomaly detection results

### Model Functions
- `create_autoencoder_model()`: Define the LSTM autoencoder architecture
- `train_model()`: Train the autoencoder model
- `save_model()`: Save the trained model and scaler

### Anomaly Detection Functions
- `calculate_threshold()`: Determine anomaly detection threshold
- `detect_anomalies()`: Perform anomaly detection on test data

## Data Format

The script expects CSV data with the following columns:
- `timestamp`: Timestamp column (will be converted to datetime and used as index)
- `YT.11FI_02044.PV`: Flow sensor data
- `YT.11PIC_02044.PV`: Pressure sensor data  
- `YT.11TI_02044.PV`: Temperature sensor data

Additional sensor columns are supported and will be automatically detected.

## Outputs

The script generates the following outputs:

1. **Model Files**:
   - `sensor_anomaly_model.h5`: Trained LSTM autoencoder model
   - `sensor_scaler.pkl`: Fitted StandardScaler for data normalization

2. **Result Files**:
   - `anomaly_results.csv`: Complete results with loss values and anomaly flags

3. **Visualization Files** (if not using `--no-plots`):
   - `sensor_data.png`: Time series plot of sensor data
   - `train_frequency.png`: Frequency domain plot of training data
   - `test_frequency.png`: Frequency domain plot of test data
   - `anomaly_results.png`: Anomaly detection results visualization

## Model Architecture

The LSTM autoencoder uses the following architecture:

1. **Encoder**:
   - LSTM layer (32 units, return sequences)
   - LSTM layer (4 units, no return sequences)

2. **Decoder**:
   - RepeatVector layer
   - LSTM layer (4 units, return sequences)
   - LSTM layer (32 units, return sequences)
   - TimeDistributed Dense layer (output dimension)

## Anomaly Detection Method

The script uses reconstruction error-based anomaly detection:

1. Train the autoencoder on normal sensor data
2. Calculate reconstruction loss on training data
3. Set threshold as: `mean(loss) + k * std(loss)` where k=3.0 by default
4. Flag test samples with reconstruction loss above threshold as anomalies

## Testing

Run the test script to validate functionality:

```bash
python test_script.py
```

## Original Jupyter Notebook

The original analysis is available in `Sensor Anomaly Detection.ipynb`. This script maintains all the original functionality while providing better structure and usability.