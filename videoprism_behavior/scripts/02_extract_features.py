import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from videoprism import models as vp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_videoprism_model(model_name: str = 'videoprism_public_v1_base'):
    """Loads a pre-trained VideoPrism model and its weights."""
    logging.info(f"Loading VideoPrism model: {model_name}")
    flax_model = vp.MODELS[model_name]()
    loaded_state = vp.load_pretrained_weights(model_name)
    
    @jax.jit
    def forward_fn(inputs):
        # The model returns a tuple (embeddings, intermediate_outputs), we only need the embeddings
        embeddings, _ = flax_model.apply(loaded_state, inputs, train=False)
        return embeddings

    return forward_fn

def extract_features(clips_dir: Path, features_dir: Path, model_fn, config: dict):
    """
    Extracts features from video clips and saves them.
    """
    clip_paths = sorted(list(clips_dir.glob('**/*.npy')))
    if not clip_paths:
        logging.error(f"No .npy clip files found in {clips_dir}. Did you run 01_create_clips.py?")
        return
        
    logging.info(f"Found {len(clip_paths)} clips to process.")
    
    # Create corresponding output directories
    for clip_path in clip_paths:
        relative_path = clip_path.relative_to(clips_dir)
        output_path = features_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = config['batch_size']
    pbar = tqdm(total=len(clip_paths), desc="Extracting Features", unit="clip")
    
    for i in range(0, len(clip_paths), batch_size):
        batch_paths = clip_paths[i:i+batch_size]
        
        # Load clips and normalize
        batch_clips = [np.load(p) for p in batch_paths]
        # Normalize to [0.0, 1.0]
        batch_clips_normalized = [clip.astype(np.float32) / 255.0 for clip in batch_clips]
        
        # Stack and convert to JAX array
        batch_array = jnp.asarray(np.stack(batch_clips_normalized))
        
        # Run inference
        batch_embeddings = model_fn(batch_array)
        
        # Ensure completion and move data to CPU for saving
        batch_embeddings = jax.device_get(batch_embeddings)

        # Save each embedding
        for j, path in enumerate(batch_paths):
            relative_path = path.relative_to(clips_dir)
            output_path = features_dir / relative_path
            np.save(output_path, batch_embeddings[j])
        
        pbar.update(len(batch_paths))
        
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="Extract features from video clips using a pre-trained VideoPrism model.")
    parser.add_argument("--clips-dir", type=str, default="videoprism_behavior/processed_data/clips", help="Directory containing the processed .npy clips.")
    parser.add_argument("--features-dir", type=str, default="videoprism_behavior/processed_data/features", help="Directory to save the extracted feature .npy files.")
    parser.add_argument("--model-name", type=str, default="videoprism_public_v1_base", help="Name of the VideoPrism model to use.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction. Adjust based on GPU memory.")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    features_dir = Path(args.features_dir)
    
    if not clips_dir.exists():
        logging.error(f"Clips directory not found: {clips_dir}")
        return

    features_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'batch_size': args.batch_size
    }

    # Get model
    model_fn = get_videoprism_model(args.model_name)
    
    # Run feature extraction
    extract_features(clips_dir, features_dir, model_fn, config)

    logging.info("Feature extraction process finished.")


if __name__ == '__main__':
    main() 