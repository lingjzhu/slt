#!/bin/bash

# MM-WLAuslan Dataset Download Script
# Downloads key components (RGB, Depth, Annotations) using gdown

set -u
set -o pipefail

DEST_DIR="/home/slime-base/projects/jian/islr/mmauslan"
LABELS_FOLDER_URL="https://drive.google.com/drive/folders/1fa7tu7PfNl8JVLkRa5pUAzR7uMtVkGkk?usp=drive_link"
FAILURES=0

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Starting download of MM-WLAuslan dataset components..."

require_command() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' was not found in PATH." >&2
        exit 1
    fi
}

warn_suspicious_id() {
    local file_path=$1
    local file_id=$2
    if [ ${#file_id} -lt 32 ]; then
        echo "Warning: $file_path uses a suspiciously short Google Drive file ID: $file_id" >&2
    fi
}

download_file() {
    local file_path=$1
    local file_id=$2
    local url="https://drive.google.com/uc?id=${file_id}"

    if [ -f "$file_path" ]; then
        echo "$file_path already exists, skipping."
        return 0
    fi

    warn_suspicious_id "$file_path" "$file_id"

    mkdir -p "$(dirname "$file_path")"
    echo "Downloading $file_path (ID: $file_id)..."

    if gdown --continue "$url" -O "$file_path"; then
        return 0
    fi

    rm -f "$file_path"
    echo "Failed to download $file_path." >&2
    echo "Google Drive sometimes blocks automated downloads for heavily accessed public files." >&2
    echo "Try the browser URL below, download the file manually, and place it at:" >&2
    echo "  $url" >&2
    echo "  $DEST_DIR/$file_path" >&2
    return 1
}

download_labels_folder() {
    if [ -f "Annotation/Labels_Split/Train.json" ] && [ -f "Annotation/Labels_Split/Valid.json" ]; then
        echo "Annotation label files already exist, skipping label folder download."
        return 0
    fi

    echo "Downloading label files from official folder..."
    rm -rf "Labels & Split"

    if ! gdown --continue --folder "$LABELS_FOLDER_URL" --remaining-ok; then
        rm -rf "Labels & Split"
        echo "Failed to download labels folder: $LABELS_FOLDER_URL" >&2
        return 1
    fi

    mkdir -p "Annotation/Labels_Split"
    if [ -f "Labels & Split/Train.json" ]; then
        mv -f "Labels & Split/Train.json" "Annotation/Labels_Split/Train.json"
    fi
    if [ -f "Labels & Split/Valid.json" ]; then
        mv -f "Labels & Split/Valid.json" "Annotation/Labels_Split/Valid.json"
    fi
    rm -rf "Labels & Split"
}

require_command gdown

# --- Annotations ---
mkdir -p Annotation/Dictionary_Mapping
mkdir -p Annotation/Labels_Split
mkdir -p Annotation/Pose/Train_Valid

download_file "Annotation/Dictionary_Mapping/Dictionary_Mapping.pt" "1YxDqYl3pJLddCujddtp9ZCjo6OvKaMWK" || FAILURES=$((FAILURES + 1))
download_labels_folder || FAILURES=$((FAILURES + 1))

download_file "Annotation/Pose/Train_Valid/pose_cam1.pkl" "1Po01hMGkCElGLVxrTdA0DcJs3aqd7Im4" || FAILURES=$((FAILURES + 1))
download_file "Annotation/Pose/Train_Valid/pose_cam2.pkl" "1pikotSa_rtVlajnuMUC3ykegJObnvDWH" || FAILURES=$((FAILURES + 1))
download_file "Annotation/Pose/Train_Valid/pose_cam3.pkl" "1D3xT1lnVr5aW7mEP15Cls1ZxgrvJgB7b" || FAILURES=$((FAILURES + 1))
download_file "Annotation/Pose/Train_Valid/pose_cam4.pkl" "1X1OgEgLSpScC8UGGIV9GogCVRMN6ZspZ" || FAILURES=$((FAILURES + 1))

# --- Train Data ---
mkdir -p Train/Kinect_F Train/Kinect_L Train/Kinect_R Train/RealSense_F

download_file "Train/Kinect_F/depth.zip" "1rYbeGvInW_zL-6UfJ8A3T2N_S5I7G5Z" || FAILURES=$((FAILURES + 1))
download_file "Train/Kinect_F/rgb.zip" "1T0p2P0UuT5JMDFWg2JAMdfRE7Jw" || FAILURES=$((FAILURES + 1))

download_file "Train/Kinect_L/depth.zip" "16YbeGvInW_zL-6UfJ8A3T2N_S5I7G5Z" || FAILURES=$((FAILURES + 1))
download_file "Train/Kinect_L/rgb.zip" "1p1RndX7F_5-a1K-LIsO7p2b_76vT9t7D" || FAILURES=$((FAILURES + 1))

download_file "Train/Kinect_R/depth.zip" "1mg3rfl1QXeraH1M2jiYeQ_SrJ7sbzVnp" || FAILURES=$((FAILURES + 1))
download_file "Train/Kinect_R/rgb.zip" "1be7V1FRSbuHGZyoo185pJDmquIYin00T" || FAILURES=$((FAILURES + 1))

download_file "Train/RealSense_F/depth.zip" "1ZACPbxJvNF64M_LaHJzgRXgeHcrCX5qO" || FAILURES=$((FAILURES + 1))
download_file "Train/RealSense_F/rgb.zip" "1TG4_p2-R45J9OwZEhU0rPXjfQwaD4LZu" || FAILURES=$((FAILURES + 1))

# --- Valid Data ---
mkdir -p Valid/Kinect_F Valid/Kinect_L Valid/Kinect_R Valid/RealSense_F

download_file "Valid/Kinect_F/depth.zip" "17RDobeWE9La3tppn3GsIjj_V8-VE-d_Q" || FAILURES=$((FAILURES + 1))
download_file "Valid/Kinect_F/rgb.zip" "19kOHDC8xC73Mu9sOmaaDNl0LyDxydMbv" || FAILURES=$((FAILURES + 1))

download_file "Valid/Kinect_L/depth.zip" "1DXSMGX4jfWGCR39lkwMjF_PCfNecX8wj" || FAILURES=$((FAILURES + 1))
download_file "Valid/Kinect_L/rgb.zip" "1WBmSGS8nBylHMVtLn2aWhVHIP_7KD8_4" || FAILURES=$((FAILURES + 1))

download_file "Valid/Kinect_R/depth.zip" "1YHYnVTKxFEep64TlL9gBXLnYJsqtxi_D" || FAILURES=$((FAILURES + 1))
download_file "Valid/Kinect_R/rgb.zip" "1pc4BYKTOeCt_Sm9NPTSi01y66RQ38B4r" || FAILURES=$((FAILURES + 1))

download_file "Valid/RealSense_F/depth.zip" "1g8oCGggE94-u1gK4xIH-xAt37yi6OiTZ" || FAILURES=$((FAILURES + 1))
download_file "Valid/RealSense_F/rgb.zip" "1mpPFU8qa3Ioy0myaCZUlLETKEOfLLLWj" || FAILURES=$((FAILURES + 1))

if [ "$FAILURES" -gt 0 ]; then
    echo "Finished with $FAILURES download failure(s)." >&2
    exit 1
fi

echo "Download complete or files already exist."
