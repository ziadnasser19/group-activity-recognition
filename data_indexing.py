import os
from glob import glob
import pickle

def index_frames(vid_dir, output_file):
    """Index dataset frames without loading images into memory."""
    dataset = {}  # {video_idx: {clip_id: [frame_paths]}}
    print(f"Indexing frames in: {vid_dir}")

    for video_idx in range(55):
        video_path = os.path.join(vid_dir, str(video_idx))
        if not os.path.exists(video_path):
            print(f"Warning: Video path {video_path} does not exist")
            continue
        clip_ids = sorted(
            [d for d in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, d))]
        )
        dataset[video_idx] = {}

        print(f"[VIDEO {video_idx}] Found {len(clip_ids)} clips")

        for clip_id in clip_ids:
            clip_path = os.path.join(video_path, clip_id)
            frame_paths = sorted(glob(os.path.join(clip_path, "*.jpg")))
            dataset[video_idx][clip_id] = frame_paths
            print(f"  - Clip {clip_id}: {len(frame_paths)} frames")

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✅ Frames saved to {output_file}")
    return dataset

def index_labels(vid_dir, output_file):
    """Index dataset labels from annotations."""
    dataset = {}  # {video_idx: {frame_id: {group, players}}}
    print(f"Indexing labels in: {vid_dir}")

    for video_idx in range(55):
        video_path = os.path.join(vid_dir, str(video_idx))
        annot_path = os.path.join(video_path, "annotations.txt")
        if not os.path.exists(annot_path):
            print(f"Warning: Annotation file {annot_path} does not exist")
            continue

        video_data = {}
        print(f"[VIDEO {video_idx}] Parsing {annot_path}")

        with open(annot_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            frame_id = parts[0].replace('.jpg', '')  # Remove .jpg early
            group_label = parts[1]
            player_annot = parts[2:]

            players = []
            for i in range(0, len(player_annot), 5):
                x, y, w, h, action = player_annot[i:i+5]
                players.append({
                    "action": action,
                    "bbox": (int(x), int(y), int(w), int(h))
                })

            video_data[frame_id] = {
                "group": group_label,
                "players": players
            }

        dataset[video_idx] = video_data

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✅ Labels saved to {output_file}")
    return dataset

def merge_datasets(labels_path, frames_path, output_file, train_idx, val_idx, test_idx):
    """Merge frames and labels datasets with train/val/test splits."""
    # Load datasets
    with open(labels_path, 'rb') as f:
        labels_dataset = pickle.load(f)
    print(f"✅ Loaded labels: {len(labels_dataset)} videos")

    with open(frames_path, 'rb') as f:
        images_dataset = pickle.load(f)
    print(f"✅ Loaded frame index: {len(images_dataset)} videos")

    # Create split mapping
    split_dict = {idx: 'train' for idx in train_idx}
    split_dict.update({idx: 'val' for idx in val_idx})
    split_dict.update({idx: 'test' for idx in test_idx})

    # Initialize merged dataset
    merged_dataset = {'train': [], 'val': [], 'test': []}

    # Merge data with 9-frame selection
    for video_idx in range(55):
        if video_idx not in split_dict:
            continue

        split = split_dict[video_idx]
        image_clips = set(images_dataset[video_idx].keys())
        label_frames = set(labels_dataset[video_idx].keys())
        clip_ids = sorted(image_clips & label_frames)

        for clip_id in clip_ids:
            frame_paths = images_dataset[video_idx][clip_id]
            if len(frame_paths) != 41:
                print(f"Warning: Clip {clip_id} in video {video_idx} has {len(frame_paths)} frames, expected 41")
                continue

            # Select 9 frames (frames 17 to 25)
            selected_frame_paths = frame_paths[17:26]
            if len(selected_frame_paths) != 9:
                print(f"Error: Selected {len(selected_frame_paths)} frames for clip {clip_id} in video {video_idx}")
                continue

            clip = {
                'video_idx': video_idx,
                'clip_id': clip_id,
                'frame_paths': selected_frame_paths,
                'group_label': labels_dataset[video_idx][clip_id]['group'],
                'players': labels_dataset[video_idx][clip_id]['players']
            }
            merged_dataset[split].append(clip)

    with open(output_file, 'wb') as f:
        pickle.dump(merged_dataset, f)
    print(f"✅ Merged dataset saved to {output_file}")

    # Print stats
    print(f"Train clips: {len(merged_dataset['train'])}")
    print(f"Val clips: {len(merged_dataset['val'])}")
    print(f"Test clips: {len(merged_dataset['test'])}")

    if merged_dataset['train']:
        example_clip = merged_dataset['train'][0]
        print(f"Example train clip: video_idx={example_clip['video_idx']}, clip_id={example_clip['clip_id']}, "
              f"num_frames={len(example_clip['frame_paths'])}, group_label={example_clip['group_label']}, "
              f"num_players={len(example_clip['players'])}")
    else:
        print("No train clips found. Check data alignment.")

    return merged_dataset