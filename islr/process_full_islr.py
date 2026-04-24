import os
import zipfile
import csv
import subprocess
import json
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# Paths
BASE_DIR = Path("/home/slime-base/projects/jian/islr/27261843")
PICS_DIR = BASE_DIR / "Pics"
GLOSS_FILE = BASE_DIR / "gloss.csv"
OUTPUT_DIR = Path("/home/slime-base/projects/jian/islr/preprocessed_videos_full")
OUTPUT_CSV = Path("/home/slime-base/projects/jian/islr/preprocessed_data_full.csv")

# Constants
MAX_WORKERS = 20
FPS = 25
LIMIT = None # Set to an integer for testing, e.g., 100

def process_single_clip(zip_path, member_folder, participant_id, view, video_id, target_path):
    """Worker task: Extract frames, convert to MP4, and return metadata."""
    if target_path.exists():
        # Quick check for metadata if re-running
        try:
             # We need total_frames. Since we already skipped, we might need a way to get it.
             # For simplicity in this script, we'll re-probe if needed, or just assume success.
             # But let's assume we want to skip.
             return None 
        except:
            pass

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Extract frames from zip
        # member_folder example: "Participant_01/front/0000/"
        # We use unzip -j to extract all files in that folder flattened to temp_path
        cmd_extract = [
            "unzip", "-j", str(zip_path), f"{member_folder}*.jpg", "-d", str(temp_path)
        ]
        subprocess.run(cmd_extract, capture_output=True, check=True)
        
        # 2. Count frames
        frames = sorted(list(temp_path.glob("*.jpg")))
        total_frames = len(frames)
        if total_frames == 0:
            return None
        
        # 3. Convert to MP4 using ffmpeg
        # ffmpeg -framerate 25 -i %05d.jpg -c:v libx264 -pix_fmt yuv420p target.mp4
        cmd_ffmpeg = [
            "ffmpeg", "-y", "-v", "error",
            "-framerate", str(FPS),
            "-i", str(temp_path / "%05d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(target_path)
        ]
        subprocess.run(cmd_ffmpeg, check=True)
        
        duration = total_frames / FPS
        return {
            "participant": participant_id,
            "id": video_id,
            "view": view,
            "filepath": str(target_path),
            "duration": duration,
            "total_frames": total_frames
        }

def load_gloss():
    gloss = {}
    with open(GLOSS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss[row['#ID'].strip()] = row['Chinese Sign Language Word']
    return gloss

def main():
    (OUTPUT_DIR / "front").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "left").mkdir(parents=True, exist_ok=True)
    
    gloss_map = load_gloss()
    zip_files = sorted(list(PICS_DIR.glob("Participant_*.zip")))
    
    tasks = []
    print("Indexing zip files...")
    for zip_path in zip_files:
        participant_id = zip_path.stem.split('_')[1]
        with zipfile.ZipFile(zip_path, 'r') as z:
            # We want unique folders like Participant_XX/view/ID/
            folders = set()
            for name in z.namelist():
                if name.endswith(".jpg"):
                    # Participant_01/front/0000/00001.jpg -> Participant_01/front/0000/
                    parts = name.split('/')
                    if len(parts) >= 4:
                        folders.add("/".join(parts[:3]) + "/")
            
            for folder in sorted(list(folders)):
                parts = folder.split('/')
                view = parts[1]
                video_id = parts[2]
                
                target_filename = f"P{participant_id}_{video_id}_{view}.mp4"
                target_path = OUTPUT_DIR / view / target_filename
                
                tasks.append((zip_path, folder, participant_id, view, video_id, target_path))
                if LIMIT and len(tasks) >= LIMIT:
                    break
            if LIMIT and len(tasks) >= LIMIT:
                break

    print(f"Total clips to process: {len(tasks)}")
    
    results = defaultdict(lambda: defaultdict(dict))
    
    # Process in parallel
    completed_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_clip, *task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            res = future.result()
            if res:
                results[res["participant"]][res["id"]][res["view"]] = res
            
            completed_count += 1
            if completed_count % 100 == 0:
                print(f"Progress: {completed_count}/{len(tasks)} clips processed.")

    # Write CSV
    print(f"Writing results to {OUTPUT_CSV}...")
    csv_rows = []
    # Build paired rows
    # Note: we need to handle existing files or missing views
    # For a robust CSV, we iterate through all participants and IDs found
    participant_ids = sorted(results.keys())
    for p_id in participant_ids:
        # All video IDs for this participant
        video_ids = sorted(results[p_id].keys())
        for v_id in video_ids:
            views_dict = results[p_id][v_id]
            f_res = views_dict.get("front")
            l_res = views_dict.get("left")
            
            # Transcription
            transcription = gloss_map.get(v_id, "Unknown")
            
            # Use metadata from front if available, else left
            primary = f_res if f_res else l_res
            
            csv_rows.append({
                "filepath_front": f_res["filepath"] if f_res else "",
                "filepath_left": l_res["filepath"] if l_res else "",
                "transcription": transcription,
                "id": v_id,
                "duration": primary["duration"] if primary else 0,
                "total_frames": primary["total_frames"] if primary else 0
            })

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filepath_front", "filepath_left", "transcription", "id", "duration", "total_frames"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print("Done!")

if __name__ == "__main__":
    main()
