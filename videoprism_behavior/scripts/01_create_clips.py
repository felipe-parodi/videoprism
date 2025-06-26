import argparse
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_bbox_json(bbox_path: Path) -> pd.DataFrame:
    """Parses the specific JSON format for bounding boxes."""
    logging.info(f"Parsing bounding box JSON from: {bbox_path}")
    with open(bbox_path, 'r') as f:
        data = json.load(f)
    
    records = []
    # The frame_id in the JSON is 1-based, video frames are 0-based.
    for item in data.get('instance_info', []):
        frame_id = item.get('frame_id')
        if frame_id is None:
            continue
        
        # Assuming one instance per frame for this task
        if item.get('instances'):
            instance = item['instances'][0]
            if instance.get('bbox'):
                bbox = instance['bbox'][0]  # It's nested in a list
                records.append({
                    # Convert 1-based JSON frame_id to 0-based index
                    'frame_idx': frame_id - 1,
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                })
    
    if not records:
        raise ValueError("No valid bounding box records found in JSON file.")
        
    df = pd.DataFrame(records)

    # Interpolate missing bounding boxes ---
    logging.info("Interpolating missing bounding boxes (limit: 40 frames)...")
    # Create a full index from min to max frame found in the data
    full_index = pd.Index(
        range(df['frame_idx'].min(), df['frame_idx'].max() + 1), name='frame_idx'
    )
    # Set index and reindex to create NaN rows for missing frames
    df = df.set_index('frame_idx').reindex(full_index)

    # Interpolate, but only for gaps of 40 frames or less
    df.interpolate(method='linear', limit=40, limit_direction='forward', inplace=True)


    # The DataFrame will still contain NaNs for gaps > 40,
    # which will be handled in the clip creation loop.
    return df

def parse_boris_annotations(annotations_df: pd.DataFrame, fps: float) -> list:
    """
    Parses BORIS annotations to extract continuous behavior segments.

    Args:
        annotations_df: DataFrame loaded from a BORIS csv file.
        fps: The frames per second of the video.

    Returns:
        A list of dictionaries, each representing a continuous behavior segment
        with 'behavior', 'start_frame', and 'end_frame'.
    """
    behavior_events = annotations_df[annotations_df['Behavior type'].isin(['START', 'STOP'])]
    
    active_behaviors = {}
    behavior_segments = []

    for _, row in behavior_events.iterrows():
        behavior = row['Behavior']
        time_s = row['Time']
        event_type = row['Behavior type']
        frame_idx = int(time_s * fps)

        if event_type == 'START':
            if behavior not in active_behaviors:
                active_behaviors[behavior] = frame_idx
        elif event_type == 'STOP':
            if behavior in active_behaviors:
                start_frame = active_behaviors.pop(behavior)
                behavior_segments.append({
                    'behavior': behavior,
                    'start_frame': start_frame,
                    'end_frame': frame_idx
                })
    
    # Handle behaviors that were started but not stopped by the end of the file
    if active_behaviors:
        logging.warning(f"Found unterminated behaviors: {list(active_behaviors.keys())}")
        # Optionally, you could decide to end them at the last frame of the video
        # For now, they are ignored.

    return behavior_segments

def process_segment(segment, video_path_str: str, bboxes_df: pd.DataFrame, output_dir: Path, config: dict):
    """
    Worker function to process a single behavior segment.
    This function is designed to be called by a multiprocessing pool.
    """
    video_path = Path(video_path_str)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Worker could not open video file: {video_path}")
        return []

    behavior = segment['behavior']
    start_frame = segment['start_frame']
    end_frame = segment['end_frame']
    local_annotations = []

    for clip_start_frame in range(start_frame, end_frame - config['window_size'] + 1, config['stride']):
        clip_end_frame = clip_start_frame + config['window_size']
        
        try:
            clip_bboxes = bboxes_df.loc[clip_start_frame:clip_end_frame-1]
        except KeyError:
             continue # Skip if frames are out of bounds

        if clip_bboxes.isnull().values.any() or len(clip_bboxes) != config['window_size']:
            continue
    
        clip_frames = []
        for frame_idx in range(clip_start_frame, clip_end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            bbox = clip_bboxes.loc[frame_idx]

            if config['flip']:
                frame = cv2.flip(frame, 0)
                x1_orig, y1_orig, x2_orig, y2_orig = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                y1_flipped, y2_flipped = h - y2_orig, h - y1_orig
                x1, y2, x2, y1 = x1_orig, y2_flipped, x2_orig, y1_flipped
            else:
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])

            pad_w = (x2 - x1) * config['bbox_padding'] / 2
            pad_h = (y2 - y1) * config['bbox_padding'] / 2
            x1, y1 = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
            x2, y2 = min(w, int(x2 + pad_w)), min(h, int(y2 + pad_h))
            
            cropped_frame = frame[y1:y2, x1:x2]
            resized_frame = cv2.resize(cropped_frame, (config['resize_dim'], config['resize_dim']))
            
            # --- ADDED: Draw debug info onto the frame ---
            # Draw a border to represent the bounding box crop
            cv2.rectangle(resized_frame, (2, 2), (config['resize_dim'] - 3, config['resize_dim'] - 3), (0, 255, 0), 2)
            # Draw the behavior label and frame number
            label_text = f"{behavior} (F: {frame_idx})"
            cv2.putText(resized_frame, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # --- END ADDED ---

            clip_frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        
        if len(clip_frames) == config['window_size']:
            clip_array = np.array(clip_frames)
            clip_filename = f"{video_path.stem}_{behavior}_{clip_start_frame}.npy"
            clip_output_path = output_dir / behavior / clip_filename
            np.save(clip_output_path, clip_array)
            local_annotations.append({
                'clip_path': str(clip_output_path.relative_to(output_dir.parent)),
                'label': behavior,
            })

    cap.release()
    return local_annotations

def create_clips(video_path: Path, bboxes_df: pd.DataFrame, behavior_segments: list, output_dir: Path, config: dict):
    """
    Extracts, processes, and saves video clips in parallel using multiprocessing.
    """
    # --- MODIFICATION: Visualization Sanity Check ---
    if config['visualize']:
        logging.info("Running visualization sanity check...")
        visualize_first_clip(video_path, bboxes_df, behavior_segments, config)
        logging.info("Visualization finished. Press any key in the popup window to continue.")
    # --- END MODIFICATION ---

    # Ensure output directories exist for all behaviors before starting workers
    for segment in behavior_segments:
        (output_dir / segment['behavior']).mkdir(parents=True, exist_ok=True)
        
    num_workers = config['num_workers']
    
    # --- OPTIMIZATION: Sort segments to make video reads more sequential ---
    sorted_segments = sorted(behavior_segments, key=lambda x: x['start_frame'])

    # Use a partial function to pass the "fixed" arguments to the worker
    worker_func = partial(
        process_segment,
        video_path_str=str(video_path),
        bboxes_df=bboxes_df,
        output_dir=output_dir,
        config=config
    )

    if num_workers > 1:
        logging.info(f"Starting clip creation with {num_workers} worker processes.")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_func, sorted_segments), total=len(sorted_segments), desc="Processing Segments"))
    else:
        logging.info("Running in single-threaded mode for easier debugging.")
        results = []
        for segment in tqdm(sorted_segments, desc="Processing Segments (single-threaded)"):
            results.append(worker_func(segment))


    # Flatten the list of lists into a single list of annotations
    master_annotations = [item for sublist in results for item in sublist]
    
    # Save the master annotations file
    if master_annotations:
        annotations_df = pd.DataFrame(master_annotations)
        master_csv_path = output_dir / 'annotations.csv'
        annotations_df.to_csv(master_csv_path, index=False)
        logging.info(f"Saved master annotations to {master_csv_path}")

def precompute_clip_count(behavior_segments: list, bboxes_df: pd.DataFrame, config: dict) -> int:
    """
    Calculates the exact number of clips that will be generated without processing video frames.
    """
    total_clips = 0
    window_size = config['window_size']
    stride = config['stride']

    logging.info("Pre-calculating the exact number of clips to be generated...")
    # --- OPTIMIZATION: Sort segments to improve cache locality during lookup ---
    sorted_segments = sorted(behavior_segments, key=lambda x: x['start_frame'])
    for segment in tqdm(sorted_segments, desc="Pre-calculating clips"):
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        for clip_start_frame in range(start_frame, end_frame - window_size + 1, stride):
            clip_end_frame = clip_start_frame + window_size
            
            try:
                # Check if the bounding box data for this entire clip is valid
                clip_bboxes = bboxes_df.loc[clip_start_frame:clip_end_frame-1]
                if not clip_bboxes.isnull().values.any() and len(clip_bboxes) == window_size:
                    total_clips += 1
            except KeyError:
                # This happens if a clip's frame range goes beyond the available bboxes
                continue
                
    return total_clips

def visualize_first_clip(video_path: Path, bboxes_df: pd.DataFrame, behavior_segments: list, config: dict):
    """
    Processes and displays the last frame of the first valid clip for a visual sanity check.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error("Visualize: Could not open video.")
        return

    # Find the very first valid clip we can generate
    for segment in behavior_segments:
        behavior = segment['behavior']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        for clip_start_frame in range(start_frame, end_frame - config['window_size'] + 1, config['stride']):
            # 1. Check if all bbox data for the clip is present and valid
            try:
                clip_bboxes = bboxes_df.loc[clip_start_frame : clip_start_frame + config['window_size'] - 1]
            except KeyError:
                continue
            if clip_bboxes.isnull().values.any() or len(clip_bboxes) != config['window_size']:
                continue

            # 2. We have a valid clip. Get the index of the last frame.
            last_frame_idx = clip_start_frame + config['window_size'] - 1
            
            # 3. Read the last frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Visualize: Could not read last frame ({last_frame_idx}) for clip at {clip_start_frame}. Trying next clip.")
                continue

            # 4. We read the frame. Now process and display it.
            h, w, _ = frame.shape
            bbox = clip_bboxes.loc[last_frame_idx]

            if config['flip']:
                frame = cv2.flip(frame, 0)
                x1_orig, y1_orig, x2_orig, y2_orig = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                y1_flipped, y2_flipped = h - y2_orig, h - y1_orig
                x1, y2, x2, y1 = x1_orig, y2_flipped, x2_orig, y1_flipped
            else:
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            pad_w = (x2 - x1) * config['bbox_padding'] / 2
            pad_h = (y2 - y1) * config['bbox_padding'] / 2
            final_x1, final_y1 = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
            final_x2, final_y2 = min(w, int(x2 + pad_w)), min(h, int(y2 + pad_h))

            cv2.rectangle(frame, (final_x1, final_y1), (final_x2, final_y2), (0, 255, 0), 2)
            label_text = f"Last Frame of First Clip: {behavior}"
            cv2.putText(frame, label_text, (final_x1, final_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow("Visualization Sanity Check (Last Frame)", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cap.release()
            return # We are done, only visualize one frame

    logging.warning("Visualize: Could not find any valid clips to visualize.")
    cap.release()

def save_first_valid_clip(video_path: Path, bboxes_df: pd.DataFrame, behavior_segments: list, save_path: Path, config: dict):
    """
    Finds the first valid clip, processes all its frames, and saves it as an MP4 video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Save Clip: Could not open video file: {video_path}")
        return

    for segment in behavior_segments:
        behavior = segment['behavior']
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        for clip_start_frame in range(start_frame, end_frame - config['window_size'] + 1, config['stride']):
            try:
                clip_bboxes = bboxes_df.loc[clip_start_frame : clip_start_frame + config['window_size'] - 1]
            except KeyError:
                continue
            if clip_bboxes.isnull().values.any() or len(clip_bboxes) != config['window_size']:
                continue

            # Found a valid clip, now process its frames
            clip_frames = []
            for frame_idx in range(clip_start_frame, clip_start_frame + config['window_size']):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"Save Clip: Failed to read frame {frame_idx}. Aborting debug save.")
                    cap.release()
                    return

                h, w, _ = frame.shape
                bbox = clip_bboxes.loc[frame_idx]

                if config['flip']:
                    frame = cv2.flip(frame, 0)
                    x1_orig, y1_orig, x2_orig, y2_orig = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    y1_flipped, y2_flipped = h - y2_orig, h - y1_orig
                    x1, y2, x2, y1 = x1_orig, y2_flipped, x2_orig, y1_flipped
                else:
                    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])

                pad_w = (x2 - x1) * config['bbox_padding'] / 2
                pad_h = (y2 - y1) * config['bbox_padding'] / 2
                x1, y1 = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
                x2, y2 = min(w, int(x2 + pad_w)), min(h, int(y2 + pad_h))
                
                cropped_frame = frame[y1:y2, x1:x2]
                resized_frame = cv2.resize(cropped_frame, (config['resize_dim'], config['resize_dim']))
                
                # --- ADDED: Draw debug info onto the frame ---
                # Draw a border to represent the bounding box crop
                cv2.rectangle(resized_frame, (2, 2), (config['resize_dim'] - 3, config['resize_dim'] - 3), (0, 255, 0), 2)
                # Draw the behavior label and frame number
                label_text = f"{behavior} (F: {frame_idx})"
                cv2.putText(resized_frame, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # --- END ADDED ---

                clip_frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            
            # Save the collected frames to a video file
            if len(clip_frames) == config['window_size']:
                logging.info(f"Saving debug clip of {config['window_size']} frames to {save_path}...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(save_path), fourcc, 10.0, (config['resize_dim'], config['resize_dim']))
                
                if not out.isOpened():
                    logging.error(f"Could not open video writer for path: {save_path}. Check OpenCV logs for more details.")
                    cap.release()
                    return
                
                for frame_rgb in clip_frames:
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                logging.info("Debug clip saved successfully.")
                cap.release()
                return

    logging.warning("Save Clip: Could not find any valid clips to save.")
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Process raw video and annotations to create labeled clips.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to the raw video file.")
    parser.add_argument("--labels-path", type=str, required=True, help="Path to the BORIS-exported CSV labels file.")
    parser.add_argument("--bboxes-path", type=str, required=True, help="Path to the JSON file with bounding boxes.")
    parser.add_argument("--output-dir", type=str, default="videoprism_behavior/processed_data/clips", help="Directory to save the processed clips and annotations.")
    parser.add_argument("--window-size", type=int, default=16, help="Number of frames per clip.")
    parser.add_argument("--stride", type=int, default=8, help="Stride for the sliding window.")
    parser.add_argument("--bbox-padding", type=float, default=0.3, help="Percentage of padding to add to bounding boxes.")
    parser.add_argument("--resize-dim", type=int, default=288, help="Dimension to resize cropped frames to (e.g., 288 for 288x288).")
    parser.add_argument("--flip", action="store_true", help="Vertically flip the video and bounding boxes if they are upside-down.")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 2), help="Number of CPU cores to use for processing. Set to 1 to disable multiprocessing for debugging.")
    parser.add_argument("--visualize", action="store_true", help="Display the last frame of the first valid clip for a sanity check.")
    parser.add_argument("--save-first-clip-path", type=str, default=None, help="Path to save the first valid processed clip as an MP4 for debugging. If set, the script will exit after saving.")
    args = parser.parse_args()

    # Prepare paths
    video_path = Path(args.video_path)
    labels_path = Path(args.labels_path)
    bboxes_path = Path(args.bboxes_path)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading annotations and bounding boxes...")
    try:
        annotations_df = pd.read_csv(labels_path)
        bboxes_df = parse_bbox_json(bboxes_path)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error loading input files: {e}")
        return

    # Get video metadata
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {args.video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    config = {
        'window_size': args.window_size,
        'stride': args.stride,
        'bbox_padding': args.bbox_padding,
        'resize_dim': args.resize_dim,
        'flip': args.flip,
        'num_workers': args.num_workers,
        'visualize': args.visualize,
    }

    # Process data
    logging.info("Parsing BORIS annotations...")
    behavior_segments = parse_boris_annotations(annotations_df, fps)
    logging.info(f"Found {len(behavior_segments)} continuous behavior segments.")
    
    # Pre-compute and report the exact number of clips
    total_clips_to_generate = precompute_clip_count(behavior_segments, bboxes_df, config)
    logging.info(f"This will generate a total of {total_clips_to_generate} valid clips.")

    if args.save_first_clip_path:
        save_path = Path(args.save_first_clip_path)
        
        # If the user provides a directory, append a default filename.
        if save_path.is_dir():
            save_path = save_path / "debug_clip.mp4"
        
        # Ensure parent directory exists before saving
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Attempting to save the first valid clip to: {save_path}")
        save_first_valid_clip(video_path, bboxes_df, behavior_segments, save_path, config)
        logging.info("Debug clip saving process finished. Exiting.")
        return

    logging.info("Starting clip creation process...")
    create_clips(video_path, bboxes_df, behavior_segments, output_dir, config)

    logging.info("Clip creation process finished.")


if __name__ == '__main__':
    main() 