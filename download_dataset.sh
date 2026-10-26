#!/bin/bash
# Script to download the CIFAR-10 and Tiny ImageNet datasets once to NFS storage
# (FEMNIST is not downloaded by this script, you need to get it from LEAF)
# Run this once before running experiments

NFS_DIR=... # Absolute path to the NFS directory where the code is stored (e.g., /nfs/ARGUS)
DATA_DIR=$NFS_DIR/eval/data

echo "==> Downloading CIFAR-10 and Tiny ImageNet to NFS"
echo "  Data directory: $DATA_DIR"

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Python environment
env_python=...

# Run the download script
cd $NFS_DIR
PYTHONPATH=src $env_python download_dataset.py

echo "==> Dataset download complete"
echo "  CIFAR-10 and Tiny ImageNet saved to: $DATA_DIR"
