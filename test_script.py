#!/usr/bin/env python3
"""
Test script for sensor_anomaly_detection.py

This script tests the main functionality of the converted Python script.
"""

import os
import sys
import subprocess
import tempfile
import shutil

def test_script_functionality():
    """Test the main script functionality"""
    print("Testing sensor_anomaly_detection.py...")
    
    # Create a temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Check if script runs with --help
        print("Test 1: Help command...")
        result = subprocess.run([
            sys.executable, "sensor_anomaly_detection.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Help command successful")
        else:
            print("✗ Help command failed")
            return False
        
        # Test 2: Run script with minimal epochs and no plots
        print("Test 2: Running with minimal configuration...")
        result = subprocess.run([
            sys.executable, "sensor_anomaly_detection.py",
            "--data-path", "气化一期S4_imputed.csv",
            "--epochs", "1",
            "--no-plots",
            "--output-dir", temp_dir
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Script execution successful")
        else:
            print("✗ Script execution failed")
            print("Error:", result.stderr)
            return False
        
        # Test 3: Check if output files are created
        print("Test 3: Checking output files...")
        expected_files = [
            "sensor_anomaly_model.h5",
            "sensor_scaler.pkl",
            os.path.join(temp_dir, "anomaly_results.csv")
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} created successfully")
            else:
                print(f"✗ {file_path} not found")
                return False
        
        print("All tests passed! ✓")
        return True
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        # Clean up generated files
        cleanup_files = [
            "sensor_anomaly_model.h5",
            "sensor_scaler.pkl"
        ]
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    success = test_script_functionality()
    sys.exit(0 if success else 1)