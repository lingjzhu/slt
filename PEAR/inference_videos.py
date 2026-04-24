import argparse
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import decord
from huggingface_hub import hf_hub_download
from pytorch3d.renderer import PointLights
from scipy.signal import savgol_filter

from models.modules.ehm import EHM_v2
from models.modules.renderer.body_renderer import Renderer2 as BodyRenderer
from models.pipeline.ehm_pipeline import Ehm_Pipeline
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.graphics_utils import GS_Camera
from utils.pipeline_utils import to_tensor


def pad_and_resize(img, target_size=512):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    return padded_img


def build_cameras_kwargs(batch_size, focal_length, device):
    screen_size = (
        torch.tensor([1024, 1024], device=device).float()[None].repeat(batch_size, 1)
    )
    return {
        "principal_point": torch.zeros(batch_size, 2, device=device).float(),
        "focal_length": focal_length,
        "image_size": screen_size,
        "device": device,
    }


def adaptive_smooth(sequence, preferred_window, polyorder):
    seq = np.asarray(sequence.detach().cpu())
    num_frames = seq.shape[0]
    if num_frames < 3:
        return sequence

    window = min(preferred_window, num_frames if num_frames % 2 == 1 else num_frames - 1)
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return sequence

    smoothed = savgol_filter(
        seq,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    return torch.from_numpy(smoothed).to(sequence.device)


def gather_video_paths(input_path, max_videos):
    path = Path(input_path)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if path.is_file():
        return [path]

    videos = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    videos = sorted(videos)
    return videos[:max_videos] if max_videos is not None else videos


def make_preview(original_frames, mesh_frames, out_path):
    if not original_frames or not mesh_frames:
        return

    indices = sorted(set([0, len(mesh_frames) // 2, len(mesh_frames) - 1]))
    panels = []
    for idx in indices:
        orig = pad_and_resize(original_frames[idx], target_size=384)
        mesh = pad_and_resize(mesh_frames[idx], target_size=384)
        panel = np.concatenate([orig, mesh], axis=1)
        cv2.putText(
            panel,
            f"frame {idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    preview = np.concatenate(panels, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


class PearVideoInferencer:
    def __init__(self, config_name="infer", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        meta_cfg = ConfigDict(model_config_path=os.path.join("configs", f"{config_name}.yaml"))
        meta_cfg = add_extra_cfgs(meta_cfg)

        self.body_renderer = BodyRenderer("assets/SMPLX", 1024, focal_length=24.0).to(self.device)
        self.ehm_model = Ehm_Pipeline(meta_cfg).to(self.device)
        self.ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(self.device)
        self.lights = PointLights(device=self.device, location=[[0.0, -1.0, -10.0]])

        repo_id = "BestWJH/PEAR_models"
        filename = "ehm_model_stage1.pt"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.ehm_model.backbone.load_state_dict(state["backbone"], strict=False)
        self.ehm_model.head.load_state_dict(state["head"], strict=False)
        self.ehm_model.eval()
        self.ehm.eval()

    @torch.no_grad()
    def process_video(self, video_path, output_dir, clip_seconds=3.0):
        output_dir.mkdir(parents=True, exist_ok=True)
        reader = decord.VideoReader(str(video_path))
        fps = float(reader.get_avg_fps()) if reader.get_avg_fps() else 30.0
        max_frames = min(len(reader), max(1, int(round(fps * clip_seconds))))

        original_frames = []
        body_sequence = []
        flame_sequence = []
        cam_sequence = []

        for idx in range(max_frames):
            frame = reader[idx].asnumpy()
            original_frames.append(frame)
            resized = pad_and_resize(frame, target_size=256)
            img_patch = to_tensor(resized, self.device)
            img_patch = torch.permute(img_patch / 255, (2, 0, 1)).unsqueeze(0)
            outputs = self.ehm_model(img_patch)
            body_sequence.append(outputs["body_param"])
            flame_sequence.append(outputs["flame_param"])
            cam_sequence.append(outputs["pd_cam"])

        fields1 = [
            "global_pose",
            "body_pose",
            "left_hand_pose",
            "right_hand_pose",
            "hand_scale",
            "head_scale",
            "exp",
            "shape",
        ]
        processed1 = {}
        for key in fields1:
            data_tensor = torch.cat([seq[key] for seq in body_sequence], dim=0)
            processed1[key] = adaptive_smooth(data_tensor, preferred_window=7, polyorder=2)

        fields2 = [
            "eye_pose_params",
            "pose_params",
            "jaw_params",
            "eyelid_params",
            "expression_params",
            "shape_params",
        ]
        processed2 = {}
        for key in fields2:
            data_tensor = torch.cat([seq[key] for seq in flame_sequence], dim=0)
            processed2[key] = adaptive_smooth(data_tensor, preferred_window=5, polyorder=2)

        cam_sequence = torch.cat(cam_sequence, dim=0)
        cam_sequence = adaptive_smooth(cam_sequence, preferred_window=7, polyorder=2)

        mesh_frames = []
        vertices_list = []
        for idx in range(max_frames):
            body_dict = {
                "global_pose": processed1["global_pose"][idx:idx + 1],
                "body_pose": processed1["body_pose"][idx:idx + 1],
                "left_hand_pose": processed1["left_hand_pose"][idx:idx + 1],
                "right_hand_pose": processed1["right_hand_pose"][idx:idx + 1],
                "hand_scale": processed1["hand_scale"][idx:idx + 1],
                "head_scale": processed1["head_scale"][idx:idx + 1],
                "exp": processed1["exp"][idx:idx + 1],
                "shape": processed1["shape"][idx:idx + 1],
                "eye_pose": None,
                "jaw_pose": None,
                "joints_offset": None,
            }
            flame_dict = {
                "eye_pose_params": processed2["eye_pose_params"][idx:idx + 1],
                "pose_params": processed2["pose_params"][idx:idx + 1],
                "jaw_params": processed2["jaw_params"][idx:idx + 1],
                "eyelid_params": processed2["eyelid_params"][idx:idx + 1],
                "expression_params": processed2["expression_params"][idx:idx + 1],
                "shape_params": processed2["shape_params"][idx:idx + 1],
            }

            pd_cam = cam_sequence[idx:idx + 1]
            pd_smplx_dict = self.ehm(body_dict, flame_dict, pose_type="aa")
            pd_camera = GS_Camera(
                **build_cameras_kwargs(1, 24, self.device),
                R=pd_cam[:, :3, :3],
                T=pd_cam[:, :3, 3],
            )
            pd_mesh_img = self.body_renderer.render_mesh(
                pd_smplx_dict["vertices"][None, 0, ...],
                pd_camera,
                lights=self.lights,
            )
            pd_mesh_img = (
                pd_mesh_img[:, :3].detach().cpu().numpy().clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
            )
            mesh_frames.append(pd_mesh_img)
            vertices_list.append(pd_smplx_dict["vertices"][0].detach().cpu().numpy())

        mesh_video_path = output_dir / "mesh_video.mp4"
        writer = imageio.get_writer(
            mesh_video_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "faststart"],
            macro_block_size=None,
        )
        for img in mesh_frames:
            h, w = img.shape[:2]
            writer.append_data(img[: h - (h % 2), : w - (w % 2)])
        writer.close()

        faces = self.body_renderer.faces[0].detach().cpu().numpy()
        vertices = np.stack(vertices_list, axis=0)
        np.savez_compressed(output_dir / "results.npz", vertices=vertices, faces=faces)
        make_preview(original_frames, mesh_frames, output_dir / "preview.jpg")

        return {
            "video_path": str(video_path),
            "fps": fps,
            "frames": max_frames,
            "mesh_video": str(mesh_video_path),
            "results_npz": str(output_dir / "results.npz"),
            "preview": str(output_dir / "preview.jpg"),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Video file or directory of videos")
    parser.add_argument("--output_root", required=True, help="Directory for per-video outputs")
    parser.add_argument("--max_videos", type=int, default=3)
    parser.add_argument("--clip_seconds", type=float, default=3.0)
    parser.add_argument("--config_name", default="infer")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    videos = gather_video_paths(args.input_path, args.max_videos)
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.input_path}")

    inferencer = PearVideoInferencer(config_name=args.config_name, device=args.device)
    print(f"Found {len(videos)} video(s)")
    for video_path in videos:
        video_name = video_path.stem
        print(f"Processing {video_path}")
        outputs = inferencer.process_video(
            video_path=video_path,
            output_dir=Path(args.output_root) / video_name,
            clip_seconds=args.clip_seconds,
        )
        for key, value in outputs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
