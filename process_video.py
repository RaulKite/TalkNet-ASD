#! /usr/bin/env python3

import os
import pickle
from ultralytics import YOLO
import cv2
import numpy as np
#from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip
import subprocess
import os
import argparse

def process_video(video_path: str) -> tuple[str, str]:
    """
    Process video through ASD pipeline.
    
    Args:
        video_path: Path to input video
        
    Returns:
        tuple: Path to scores and tracks pickle files
    """
    # Run ASD script
    os.system(f"bash ./run_asd_for_custom_video_in_local.sh {video_path}")
    
    # Get case ID and output directory
    case_id = os.path.basename(video_path).split('.mp4')[0]
    print(50*'-')
    print(case_id)
    print(50*'-')
    output_dir = f"./output/{case_id}"
    
    return f"{output_dir}/scores.pckl", f"{output_dir}/tracks.pckl"

def analyze_scene_and_scores(csv_path: str, scores_path: str, frame_num: int = 50) -> tuple[bool, float]:
    """
    Analyze if video has single scene and get mean scores.
    
    Args:
        csv_path: Path to scenes CSV
        scores_path: Path to scores pickle file
        
    Returns:
        tuple: (has_single_scene, mean_score)
    """
    # Check if single scene
    with open(csv_path, 'r') as f:
        has_single_scene = sum(1 for line in f) - 2 == 1
        
    # Get mean scores for frames 45-55
    with open(scores_path, 'rb') as f:
        scores = pickle.load(f)
    mean_score = np.mean(scores[0][frame_num-5:frame_num+5])
    
    return has_single_scene, mean_score

def get_bbox_center(tracks_path: str, frame_num: int = 50) -> tuple[float, float]:
    """
    Get center coordinates of bbox at specified frame.
    
    Args:
        tracks_path: Path to tracks pickle file
        frame_num: Frame number to analyze
        
    Returns:
        tuple: (center_x, center_y)
    """
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    
    # Get bbox coordinates for frame 50
    frame_idx = np.where(tracks[0]['track']['frame'] == frame_num)[0][0]
    bbox = tracks[0]['track']['bbox'][frame_idx]
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    return center_x, center_y

def create_tracked_person_video(video_path: str, results: list, track_id: int, output_path: str, keep_audio: bool = True) -> None:
    """
    Creates a video with black background except for the tracked person's bounding box region.
    The bounding box is expanded by 5% on each side.
    
    Args:
        video_path: Path to the input video file
        results: List of detection results for all frames
        track_id: Track ID of the person to highlight
        output_path: Path where the output video will be saved
        keep_audio: Whether to keep the original audio in output video
        
    Returns:
        None
    """
    print("Starting video creation process...")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    print(f"Opened video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
    
    # Create absolute paths
    abs_output_path = os.path.abspath(output_path)
    # Remove .mp4 extension if present to avoid double extension
    abs_output_path = abs_output_path.replace('.mp4', '')
    temp_output = abs_output_path + '_temp.mp4'
    final_output = abs_output_path + '.mp4'
    print(f"Temporary output path: {temp_output}")
    
    # Create VideoWriter object with more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using MP4V codec instead of H.264
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("First codec failed, trying alternative...")
        # Try alternative codec if first attempt fails
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to create output video file at {temp_output}")
    
    print("Processing frames...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create black frame
        black_frame = np.zeros_like(frame)
            
        # Get detection results for current frame
        if frame_idx < len(results):
            frame_results = results[frame_idx]
            boxes = frame_results.boxes.xyxy
            ids = frame_results.boxes.id
            
            # Find tracked person and copy their region from original frame
            for box, id in zip(boxes, ids):
                if int(id) == track_id:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Calculate 5% of width and height for expansion
                    w_expand = int((x2 - x1) * 0.05)
                    h_expand = int((y2 - y1) * 0.05)
                    
                    # Expand box by 5% each side, but stay within frame bounds
                    x1 = max(0, x1 - w_expand)
                    y1 = max(0, y1 - h_expand)
                    x2 = min(width, x2 + w_expand)
                    y2 = min(height, y2 + h_expand)
                    
                    # Copy the person region from original frame to black frame
                    black_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        # Write frame
        out.write(black_frame)
        frame_idx += 1
        
    print(f"Processed {frame_idx} frames")
    
    # Release everything
    cap.release()
    out.release()

    if keep_audio:
        # Verify temp file exists
        print("OUTPUT:", 50*'-')
        print(temp_output)
        print(50*'-')
        print("ORIGINAL:", 50*'-')
        print(video_path)
        print(50*'-')
        print("FINAL OUTPUT:", 50*'-')
        print(final_output)
        print(50*'-')
        if not os.path.exists(temp_output):
            raise FileNotFoundError(f"Temporary video file not found at {temp_output}")
            
        # Extract audio from original video
        audio_path = os.path.join(os.path.dirname(temp_output), "temp_audio.aac")
        extract_audio_cmd = f'ffmpeg -i "{video_path}" -vn -acodec copy "{audio_path}" -y '
        print(extract_audio_cmd)        
        subprocess.run(extract_audio_cmd, shell=True, check=True)
            
        # Combine temp video with extracted audio
        combine_cmd = f'ffmpeg -i "{temp_output}" -i "{audio_path}" -c:v h264 -c:a aac "{final_output}" -y'
        print(combine_cmd)

        subprocess.run(combine_cmd, shell=True, check=True)
        
        # Clean up temp audio file
        # os.remove(audio_path)
            
    else:
        # Just rename temp file to final output
        os.rename(temp_output, final_output)

def get_track_id_at_point(frame_num: int, x: int, y: int, results: list) -> int | None:
    """
    Get the track ID of a person at a specific point in a frame.
    
    Args:
        frame_num: Frame number to check
        x: X coordinate of the point
        y: Y coordinate of the point 
        results: List of detection results for all frames
        
    Returns:
        Track ID if a person is found at the point, None otherwise
    """
    # Get frame results
    frame_results = results[frame_num]

    # Get all bounding boxes and IDs for this frame
    boxes = frame_results.boxes.xyxy
    track_ids = frame_results.boxes.id

    # Check each box if it contains the point
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            return int(track_id)
            
    return None

def process_one_video(video_path: str, output_dir: str = "./output", frame_num: int = 50):
    """
    Main orchestration function.
    
    Args:
        video_path: Path to input video
    """
    # Step 1: Process through ASD
    scores_path, tracks_path = process_video(video_path)
    
    # Step 2: Analyze scene and scores
    case_id = os.path.basename(video_path).split('.mp4')[0]
    print(50*'-')
    print(case_id)
    print(50*'-')
    csv_path = f"./output/{case_id}/{case_id}.csv"
    
    has_single_scene, mean_score = analyze_scene_and_scores(csv_path, scores_path, frame_num=50)
    print(f"Has single scene: {has_single_scene}")
    print(f"Mean score (frames {frame_num-5}:{frame_num+5}): {mean_score}")
    
    if has_single_scene and mean_score > 0:
        # Step 3: Get bbox center
        center_x, center_y = get_bbox_center(tracks_path, frame_num)
        print(f"Center point at frame {frame_num}: ({center_x:.2f}, {center_y:.2f})")

        model = YOLO("yolo11n.pt")
        results = model.track(source=video_path, conf=0.5, iou=0.5, show=True, save=True, save_dir="output", save_txt=True, save_json=True)
        track_id = get_track_id_at_point(frame_num, center_x, center_y, results)
        output_path = f"./output/{case_id}/{case_id}.mp4"
        
        # Step 4: Track person and create video
        create_tracked_person_video(video_path, results, track_id, output_path, keep_audio=True)

        # if output video exists, move to output folder
        if os.path.exists(output_path):
            os.rename(output_path, f"{output_dir}/{os.path.basename(output_path)}")
    else:
        print("Video doesn't meet criteria (single scene with positive mean score)")
  
    

def process_folder(input_dir: str, output_dir: str = "./salida", frame_num: int = 50) -> None:
    """
    Process all video files in a folder by calling main() on each.
    
    Args:
        input_dir: Path to folder containing video files
        output_dir: Path to output directory for processed videos
    """
    # Create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(50*'-')
        print(f"Processing: {video_file}")
        print(50*'-')
        try:
            process_one_video(video_path, output_dir=output_dir, frame_num=frame_num)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
      # Delete temporal folder
    print(50*"**")
    print(f"Deleting temporal folder: ./output")
    os.system(f"rm -rf ./output")
    # Delete runs folder if it exists
    if os.path.exists("runs"):
        os.system("rm -rf runs")
        print("Deleted runs folder")
    print(50*"**")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos through ASD pipeline and track person.")
    parser.add_argument("--input_dir", type=str, default="./input_videos", help="Path to input directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./output_videos", help="Path to output directory for processed videos")
    parser.add_argument("--second", type=int, default=2, help="Second where the word is spoken")
    args = parser.parse_args()

    # 25 fps
    frame_num = args.second * 25
    process_folder(input_dir=args.input_dir, output_dir=args.output_dir, frame_num=frame_num)


