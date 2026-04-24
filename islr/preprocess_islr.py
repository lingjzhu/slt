import os
import zipfile
import csv
import subprocess
import json
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("/home/slime-base/projects/jian/islr/27261843")
VIDEOS_DIR = BASE_DIR / "Videos"
GLOSS_FILE = BASE_DIR / "gloss.csv"
OUTPUT_DIR = Path("/home/slime-base/projects/jian/islr/preprocessed_videos")
OUTPUT_CSV = Path("/home/slime-base/projects/jian/islr/preprocessed_data.csv")

def get_video_metadata(filepath):
    """Get duration and total frames using ffprobe."""
    # Using nb_frames if available, else count_frames. But for these short videos, 
    # my test showed nb_frames works and is fast.
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=nb_frames",
        "-select_streams", "v:0",
        "-of", "json", str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        streams = data.get("streams", [])
        total_frames = int(streams[0].get("nb_frames", 0)) if streams and streams[0].get("nb_frames") else 0
        
        # Fallback if nb_frames is missing or 0
        if total_frames == 0:
            cmd_count = [
                "ffprobe", "-v", "error",
                "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "json", str(filepath)
            ]
            result_count = subprocess.run(cmd_count, capture_output=True, text=True, check=True)
            data_count = json.loads(result_count.stdout)
            streams_count = data_count.get("streams", [])
            total_frames = int(streams_count[0].get("nb_read_frames", 0)) if streams_count else 0
            
        return duration, total_frames
    except Exception as e:
        print(f"Error probing {filepath}: {e}")
        return 0.0, 0

def load_gloss():
    """Load transcription mapping from gloss.csv."""
    gloss = {}
    with open(GLOSS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_val = row['#ID'].strip()
            gloss[id_val] = row['Chinese Sign Language Word']
    return gloss

def main():
    (OUTPUT_DIR / "front").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "left").mkdir(parents=True, exist_ok=True)
    
    gloss_map = load_gloss()
    zip_files = sorted(list(VIDEOS_DIR.glob("Participant_*.zip")))
    
    # Nested dict to store paths: pairs[participant_id][video_id][view] = filepath
    pairs = defaultdict(lambda: defaultdict(dict))
    
    for zip_path in zip_files:
        participant_id = zip_path.stem.split('_')[1] # e.g. "01"
        print(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.infolist():
                if member.filename.endswith(".mp4"):
                    parts = member.filename.split('/')
                    if len(parts) >= 3:
                        view = parts[-2] # e.g. "front" or "left"
                        original_filename = parts[-1]
                        video_id = Path(original_filename).stem
                        
                        if view not in ["front", "left"]:
                            continue
                            
                        new_filename = f"P{participant_id}_{video_id}_{view}.mp4"
                        target_path = OUTPUT_DIR / view / new_filename
                        
                        # Extract
                        with z.open(member) as source, open(target_path, "wb") as target:
                            target.write(source.read())
                        
                        pairs[participant_id][video_id][view] = str(target_path)

    # Process and write CSV
    print(f"Probing metadata and writing results to {OUTPUT_CSV}...")
    csv_rows = []
    
    # Sort by participant and then ID
    for p_id in sorted(pairs.keys()):
        for v_id in sorted(pairs[p_id].keys()):
            views_dict = pairs[p_id][v_id]
            f_path = views_dict.get("front", "")
            l_path = views_dict.get("left", "")
            
            # Use front for metadata if available, else left
            primary_path = f_path if f_path else l_path
            duration, total_frames = (0.0, 0)
            if primary_path:
                duration, total_frames = get_video_metadata(primary_path)
            
            transcription = gloss_map.get(v_id, "Unknown")
            
            csv_rows.append({
                "filepath_front": f_path,
                "filepath_left": l_path,
                "transcription": transcription,
                "id": v_id,
                "duration": duration,
                "total_frames": total_frames
            })
            print(f"  Processed P{p_id} {v_id} ({transcription})")

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filepath_front", "filepath_left", "transcription", "id", "duration", "total_frames"])
        writer.writerows([{ "filepath_front": "filepath_front", "filepath_left": "filepath_left", "transcription": "transcription", "id": "id", "duration": "duration", "total_frames": "total_frames" }]) # Custom header if needed, but dictwriter handles it
        # Actually writer.writeheader() is better
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filepath_front", "filepath_left", "transcription", "id", "duration", "total_frames"])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print("Done!")

if __name__ == "__main__":
    main()
