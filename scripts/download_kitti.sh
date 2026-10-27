#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# Configuration & Directory Setup
# ==========================================
TARGET_DIR="data/KITTIDataset"

echo "=========================================="
echo " Starting KITTI 3D Dataset Setup "
echo "=========================================="

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

# Check for required command-line tools
for cmd in wget unzip; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: Required tool '$cmd' is not installed." >&2
        exit 1
    fi
done

# ==========================================
# Download Functions
# ==========================================

# 1. Left Color Images (~12 GB)
echo "[1/3] Downloading Left Color Images (data_object_image_2.zip - ~12 GB)..."
# Replace URL below with the official KITTI link
wget -c "http://www.cvlibs.net/download.php?file=data_object_image_2.zip" -O data_object_image_2.zip

# 2. Camera Calibration Matrices (~16 MB)
echo "[2/3] Downloading Camera Calibration Matrices (data_object_calib.zip - ~16 MB)..."
# Replace URL below with the official KITTI link
wget -c "http://www.cvlibs.net/download.php?file=data_object_calib.zip" -O data_object_calib.zip

# 3. Training Labels (~5 MB)
echo "[3/3] Downloading Training Labels (data_object_label_2.zip - ~5 MB)..."
# Replace URL below with the official KITTI link
wget -c "http://www.cvlibs.net/download.php?file=data_object_label_2.zip" -O data_object_label_2.zip

# ==========================================
# Extraction & Cleanup
# ==========================================

echo "=========================================="
echo " Unzipping Files... "
echo "=========================================="

echo "Extracting Left Color Images..."
unzip -q data_object_image_2.zip

echo "Extracting Calibration Matrices..."
unzip -q data_object_calib.zip

echo "Extracting Training Labels..."
unzip -q data_object_label_2.zip

echo "Cleaning up archive files to free disk space..."
rm -f *.zip

echo "=========================================="
echo " KITTI Dataset Setup Completed Successfully! "
echo "=========================================="