import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T

class VolleyballDataset(Dataset):
    def __init__(self, split_data, transform=None, person_level=False, single_frame=False):
        self.data = split_data
        self.transform = transform
        self.person_level = person_level
        self.single_frame = single_frame

        # Create label mappings
        self.group_to_idx = {label: idx for idx, label in enumerate(sorted({clip['group_label'] for clip in split_data}))}
        self.action_to_idx = {action: idx for idx, action in enumerate(sorted({player['action'] for clip in split_data for player in clip['players']}))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip = self.data[idx]
        frames = [cv2.imread(path) for path in clip['frame_paths']]
        frames = [f for f in frames if f is not None]
        if len(frames) != 9:
            raise ValueError(f"Expected 9 frames, got {len(frames)} for clip {clip['clip_id']} in video {clip['video_idx']}")

        # Convert BGR → RGB
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        # Apply transforms
        if self.transform:
            frames = [self.transform(f) for f in frames]

        group_label = self.group_to_idx[clip['group_label']]
        players = [
            {"action": self.action_to_idx[p["action"]], "bbox": p["bbox"]}
            for p in clip["players"]
        ]

        if self.person_level:
            # Crop to each player's bbox
            player_crops = []
            for player in players:
                bbox = player["bbox"]
                player_frames = []
                selected_frames = [frames[4]] if self.single_frame else frames
                for f in selected_frames:
                    f = T.ToPILImage()(f)  # Convert to PIL for crop
                    crop = f.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                    crop = self.transform(crop) if self.transform else T.ToTensor()(crop)
                    player_frames.append(crop)
                player_frames = torch.stack(player_frames)  # (T, C, H, W)
                player_crops.append(player_frames)

            return {
                "player_crops": player_crops,
                "group_label": group_label,
                "players": players,
                "video_idx": clip["video_idx"],
                "clip_id": clip["clip_id"],
            }
        else:
            # Full-frame mode
            if self.single_frame:
                frame = frames[4].unsqueeze(0)  # (1, C, H, W)
                return {"frames": frame, "group_label": group_label}
            else:
                frames = torch.stack(frames)  # (T, C, H, W)
                return {"frames": frames, "group_label": group_label}