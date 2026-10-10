import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

def interpolate_tracked_detections(input_csv, output_dir, max_gap=150):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct output file path: <output_dir>/<original_filename>_interpolated.csv
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    output_csv = output_dir / f"{base_name}_interpolated.csv"

    # Load the tracked detections
    df = pd.read_csv(input_csv)

    # Ensure data is sorted by track and chronological order
    df = df.sort_values(by=['tracker_id', 'frame']).reset_index(drop=True)

    interpolated_tracks = []

    # Process each unique track independently
    for track_id, track_df in df.groupby('tracker_id'):
        # Set 'frame' as the index to easily reindex missing chronological gaps
        track_df = track_df.set_index('frame')

        # Determine the track's full active lifespan
        min_frame = track_df.index.min()
        max_frame = track_df.index.max()
        full_frame_range = np.arange(min_frame, max_frame + 1)

        # Reindex introduces missing frames as NaN rows
        track_df_reindexed = track_df.reindex(full_frame_range)

        # Linearly interpolate bounding box corners and confidence score
        # The 'limit' parameter ensures we don't interpolate gaps larger than the tracking buffer
        cols_to_interpolate = ['x1', 'y1', 'x2', 'y2', 'confidence']
        track_df_reindexed[cols_to_interpolate] = track_df_reindexed[cols_to_interpolate].interpolate(
            method='linear', 
            limit=max_gap, 
            limit_direction='forward'
        )

        # Drop any frames that remain NaN (gaps larger than max_gap)
        track_df_reindexed = track_df_reindexed.dropna(subset=['x1'])

        # Forward/backward fill static categorical labels
        track_df_reindexed['class_id'] = track_df_reindexed['class_id'].ffill().bfill()
        track_df_reindexed['class_name'] = track_df_reindexed['class_name'].ffill().bfill()
        track_df_reindexed['tracker_id'] = track_id

        # Flag interpolated frames with a det_index of -1 for easy filtering later
        track_df_reindexed['det_index'] = track_df_reindexed['det_index'].fillna(-1)

        # Recompute center coordinates and dimensions from the new interpolated corners
        track_df_reindexed['cx'] = (track_df_reindexed['x1'] + track_df_reindexed['x2']) / 2
        track_df_reindexed['cy'] = (track_df_reindexed['y1'] + track_df_reindexed['y2']) / 2
        track_df_reindexed['width'] = track_df_reindexed['x2'] - track_df_reindexed['x1']
        track_df_reindexed['height'] = track_df_reindexed['y2'] - track_df_reindexed['y1']

        # Interpolate processed_frame (to keep it linear) and reset the index to restore 'frame' as a column
        if 'processed_frame' in track_df_reindexed.columns:
            track_df_reindexed['processed_frame'] = track_df_reindexed['processed_frame'].interpolate(method='linear')
        
        track_df_reindexed = track_df_reindexed.reset_index().rename(columns={'index': 'frame'})
        
        # Re-calculate the time column assuming a constant framerate (25fps = 0.04s based on the dataset)
        track_df_reindexed['time_s'] = track_df_reindexed['frame'] * 0.04

        interpolated_tracks.append(track_df_reindexed)

    # Reassemble all tracks into a single DataFrame
    out_df = pd.concat(interpolated_tracks, ignore_index=True)

    # Sort chronologically by frame, then by tracker_id
    out_df = out_df.sort_values(by=['frame', 'tracker_id']).reset_index(drop=True)

    # Save to the specified output path
    out_df.to_csv(output_csv, index=False)
    print(f"Interpolation complete. Saved {len(out_df) - len(df)} new frames to {output_csv}.")
    return output_csv