import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import decord
from scipy.signal import savgol_filter

from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.pipeline_utils import to_tensor


def pad_and_resize(img, target_size=256):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    return padded_img


def gather_video_paths(input_path, max_videos):
    path = Path(input_path)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if path.is_file():
        return [path]
    videos = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    videos = sorted(videos)
    return videos[:max_videos] if max_videos is not None else videos


def adaptive_smooth(sequence, preferred_window, polyorder):
    seq = np.asarray(sequence)
    num_frames = seq.shape[0]
    if num_frames < 3:
        return seq
    window = min(preferred_window, num_frames if num_frames % 2 == 1 else num_frames - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return seq
    return savgol_filter(seq, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def tensor_to_numpy_dict(data):
    out = {}
    for key, value in data.items():
        if value is None:
            continue
        out[key] = value.detach().cpu().numpy()
    return out


class PearFeatureExtractor:
    def __init__(self, config_name="infer", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta_cfg = ConfigDict(model_config_path=os.path.join("configs", f"{config_name}.yaml"))
        meta_cfg = add_extra_cfgs(meta_cfg)
        self.model = Ehm_Pipeline(meta_cfg).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_video(self, video_path, output_dir, clip_seconds=3.0, smooth_decoded=False, batch_size=1):
        output_dir.mkdir(parents=True, exist_ok=True)
        reader = decord.VideoReader(str(video_path))
        fps = float(reader.get_avg_fps()) if reader.get_avg_fps() else 30.0
        max_frames = min(len(reader), max(1, int(round(fps * clip_seconds))))

        raw_feature_frames = []
        body_param_frames = []
        flame_param_frames = []
        pd_cam_frames = []

        for start in range(0, max_frames, batch_size):
            end = min(start + batch_size, max_frames)
            frame_indices = list(range(start, end))
            frames = reader.get_batch(frame_indices).asnumpy()
            batch = []
            for frame in frames:
                resized = pad_and_resize(frame, target_size=256)
                img_patch = to_tensor(resized, self.device)
                batch.append(torch.permute(img_patch / 255, (2, 0, 1)))
            img_batch = torch.stack(batch, dim=0)
            outputs = self.model(img_batch)

            raw_feature_frames.append(tensor_to_numpy_dict(outputs["raw_features"]))
            body_param_frames.append(tensor_to_numpy_dict(outputs["body_param"]))
            flame_param_frames.append(tensor_to_numpy_dict(outputs["flame_param"]))
            pd_cam_frames.append(outputs["pd_cam"].detach().cpu().numpy())

        feature_names = sorted(raw_feature_frames[0].keys())
        raw_features = {
            key: np.concatenate([frame[key] for frame in raw_feature_frames], axis=0)
            for key in feature_names
        }

        body_names = sorted(body_param_frames[0].keys())
        body_params = {
            key: np.concatenate([frame[key] for frame in body_param_frames], axis=0)
            for key in body_names
        }

        flame_names = sorted(flame_param_frames[0].keys())
        flame_params = {
            key: np.concatenate([frame[key] for frame in flame_param_frames], axis=0)
            for key in flame_names
        }

        pd_cam = np.concatenate(pd_cam_frames, axis=0)

        if smooth_decoded:
            smooth_windows = {
                "global_pose": 7,
                "body_pose": 7,
                "left_hand_pose": 7,
                "right_hand_pose": 7,
                "hand_scale": 7,
                "head_scale": 7,
                "exp": 7,
                "shape": 7,
                "eye_pose_params": 5,
                "pose_params": 5,
                "jaw_params": 5,
                "eyelid_params": 5,
                "expression_params": 5,
                "shape_params": 5,
            }
            body_params_smoothed = {
                key: adaptive_smooth(value, preferred_window=smooth_windows[key], polyorder=2)
                for key, value in body_params.items()
            }
            flame_params_smoothed = {
                key: adaptive_smooth(value, preferred_window=smooth_windows[key], polyorder=2)
                for key, value in flame_params.items()
            }
            pd_cam_smoothed = adaptive_smooth(pd_cam, preferred_window=7, polyorder=2)
        else:
            body_params_smoothed = None
            flame_params_smoothed = None
            pd_cam_smoothed = None

        np.savez_compressed(
            output_dir / "features_raw.npz",
            fps=np.array([fps], dtype=np.float32),
            num_frames=np.array([max_frames], dtype=np.int32),
            **{f"raw/{k}": v for k, v in raw_features.items()},
            **{f"body/{k}": v for k, v in body_params.items()},
            **{f"flame/{k}": v for k, v in flame_params.items()},
            **{"camera/pd_cam": pd_cam},
        )

        if smooth_decoded:
            np.savez_compressed(
                output_dir / "features_smoothed.npz",
                fps=np.array([fps], dtype=np.float32),
                num_frames=np.array([max_frames], dtype=np.int32),
                **{f"body/{k}": v for k, v in body_params_smoothed.items()},
                **{f"flame/{k}": v for k, v in flame_params_smoothed.items()},
                **{"camera/pd_cam": pd_cam_smoothed},
            )

        return {
            "video_path": str(video_path),
            "fps": fps,
            "frames": max_frames,
            "features_raw": str(output_dir / "features_raw.npz"),
            "features_smoothed": str(output_dir / "features_smoothed.npz") if smooth_decoded else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Video file or directory of videos")
    parser.add_argument("--output_root", required=True, help="Directory for per-video feature files")
    parser.add_argument("--max_videos", type=int, default=3)
    parser.add_argument("--clip_seconds", type=float, default=3.0)
    parser.add_argument("--config_name", default="infer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smooth_decoded", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    videos = gather_video_paths(args.input_path, args.max_videos)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.input_path}")

    extractor = PearFeatureExtractor(config_name=args.config_name, device=args.device)
    print(f"Found {len(videos)} video(s)")
    for video_path in videos:
        print(f"Processing {video_path}")
        outputs = extractor.extract_video(
            video_path=video_path,
            output_dir=Path(args.output_root) / video_path.stem,
            clip_seconds=args.clip_seconds,
            smooth_decoded=args.smooth_decoded,
            batch_size=args.batch_size,
        )
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
