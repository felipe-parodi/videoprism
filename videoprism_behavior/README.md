# VideoPrism for Animal Behavior Analysis

This project adapts the `VideoPrism` foundational model for fine-grained animal behavior analysis. The goal is to leverage a state-of-the-art, frozen video encoder to extract general-purpose features from video clips and then train a lightweight, task-specific classification head to recognize complex behaviors.

This approach follows the methodology outlined in the paper ["Video Foundation Models for Animal Behavior Analysis"](https://www.biorxiv.org/content/10.1101/2024.07.30.605655v1), aiming for an efficient and scalable pipeline.

## Workflow Overview

The analysis is structured as a multi-stage pipeline:

1.  **Clip Generation (`01_create_clips.py`):** Raw videos are processed into short, labeled `.npy` clips based on bounding box detections and behavior annotations. This step ensures that only high-quality, subject-focused data is used for feature extraction.

2.  **Feature Extraction (`02_extract_features.py`):** The pre-trained VideoPrism model is used as a frozen feature extractor to generate powerful, general-purpose embeddings for each clip.

3.  **Head Training (`03_train_head.py`):** A small, lightweight classification head (e.g., a `MultiHeadAttentionPooling` layer) is trained in PyTorch on the pre-computed features. 

4.  **Inference (`04_inference.py`):** The full pipeline, combining the frozen VideoPrism encoder and the trained classification head, is used to generate a continuous ethogram for new, unlabeled videos.

This modular structure separates the heavy-lifting of feature extraction from the more lightweight model training, creating a flexible and reproducible research workflow. 