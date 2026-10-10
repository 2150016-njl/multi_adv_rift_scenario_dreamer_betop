#!/bin/bash

# Set the base directory
BASE_DIR="$(pwd)/log"

# Recursively find all "offline-run-*" folders and sync them in parallel
find "$BASE_DIR" -type d -name "offline-run-*" | xargs -P 8 -I {} bash -c 'echo "Syncing Wandb logs in: {}"; wandb sync "{}"'

echo "All Wandb logs have been processed!"
