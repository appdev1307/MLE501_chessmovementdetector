#!/usr/bin/env python3
"""
Chess Video Analyzer - Enhanced move detection and validation
Main script for analyzing chess games in video format
"""
import cv2
import os
import sys
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict, deque
import time
import json
import numpy as np
import math

def select_video_file():
    """Open a file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Chess Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All files", "*.*")
        ]
    )
    return file_path

def box_to_chess_position(box, frame_width, frame_height):
    """Convert bounding box center to chess board position (a1-h8)"""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Normalize to 0-1
    norm_x = center_x / frame_width
    norm_y = center_y / frame_height
    
    # Convert to chess coordinates
    file_idx = int(norm_x * 8)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    
    rank_idx = int(norm_y * 8)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = 8 - rank_idx
    
    return f"{file_letter}{rank_number}"

def is_valid_chess_move(piece, from_pos, to_pos):
    """Basic chess rules validation"""
    if from_pos == to_pos:
        return False
    
    from_file, from_rank = from_pos[0], int(from_pos[1])
    to_file, to_rank = to_pos[0], int(to_pos[1])
    
    file_diff = abs(ord(to_file) - ord(from_file))
    rank_diff = abs(to_rank - from_rank)
    
    piece_type = piece.lower()
    
    # Pawn moves
    if piece_type == 'p':
        if file_diff == 0:  # Same file
            if piece.isupper():  # White pawn
                return to_rank > from_rank and rank_diff <= 2
            else:  # Black pawn
                return to_rank < from_rank and rank_diff <= 2
        elif file_diff == 1 and rank_diff == 1:  # Diagonal capture
            return True
        return False
    
    # Rook moves
    elif piece_type == 'r':
        return file_diff == 0 or rank_diff == 0
    
    # Knight moves
    elif piece_type == 'n':
        return (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)
    
    # Bishop moves
    elif piece_type == 'b':
        return file_diff == rank_diff
    
    # Queen moves
    elif piece_type == 'q':
        return file_diff == 0 or rank_diff == 0 or file_diff == rank_diff
    
    # King moves
    elif piece_type == 'k':
        return file_diff <= 1 and rank_diff <= 1
    
    return True

def calculate_move_distance(from_pos, to_pos):
    """Calculate euclidean distance between chess positions"""
    from_file, from_rank = ord(from_pos[0]) - ord('a'), int(from_pos[1]) - 1
    to_file, to_rank = ord(to_pos[0]) - ord('a'), int(to_pos[1]) - 1
    
    return math.sqrt((to_file - from_file)**2 + (to_rank - from_rank)**2)

def detect_pieces_in_frame(model, frame, conf_threshold=0.6):
    """Detect chess pieces in a single frame"""
    try:
        results = model.predict(frame, verbose=False, conf=conf_threshold, save=False)
        
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                class_names = model.names
                
                piece_positions = {}
                detection_details = []
                frame_height, frame_width = frame.shape[:2]
                
                for j, (conf, cls, box) in enumerate(zip(
                    detections.conf.cpu().numpy(), 
                    detections.cls.cpu().numpy(),
                    detections.xyxy.cpu().numpy()
                )):
                    class_name = class_names[int(cls)] if int(cls) in class_names else 'unknown'
                    
                    # Skip board detection for position tracking
                    if class_name in ['board', 'chessboards-and-pieces']:
                        continue
                        
                    if conf > 0.7:  # High confidence pieces only
                        x1, y1, x2, y2 = box.astype(int)
                        chess_pos = box_to_chess_position(box, frame_width, frame_height)
                        
                        # Store best detection per position
                        if chess_pos not in piece_positions or piece_positions[chess_pos]['confidence'] < conf:
                            piece_positions[chess_pos] = {
                                'piece': class_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            }
                        
                        detection_details.append({
                            'class': class_name,
                            'conf': conf,
                            'box': (x1, y1, x2, y2),
                            'position': chess_pos
                        })
                
                return piece_positions, detection_details
        
        return {}, []
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {}, []

def detect_moves(prev_positions, curr_positions, min_confidence=0.75, min_distance=0.4):
    """Detect moves by comparing piece positions between frames"""
    moves = []
    
    if not prev_positions:
        return moves
    
    # Track pieces that disappeared from previous positions
    for prev_pos, prev_info in prev_positions.items():
        piece_type = prev_info['piece']
        
        # Look for the same piece type in new positions
        for curr_pos, curr_info in curr_positions.items():
            if (curr_info['piece'] == piece_type and 
                curr_pos != prev_pos and
                curr_pos not in prev_positions):
                
                # Calculate move properties
                distance = calculate_move_distance(prev_pos, curr_pos)
                avg_confidence = (prev_info['confidence'] + curr_info['confidence']) / 2
                
                # Apply filters
                if (avg_confidence >= min_confidence and 
                    distance >= min_distance and
                    is_valid_chess_move(piece_type, prev_pos, curr_pos)):
                    
                    moves.append({
                        'move': f"{piece_type}{prev_pos}-{curr_pos}",
                        'piece': piece_type,
                        'from': prev_pos,
                        'to': curr_pos,
                        'confidence': avg_confidence,
                        'distance': distance
                    })
    
    return moves

def annotate_frame_with_moves(frame, detection_details, recent_moves):
    """Annotate frame with detections and recent moves"""
    annotated_frame = frame.copy()
    
    # Draw piece detections
    for detection in detection_details:
        class_name = detection['class']
        conf = detection['conf']
        x1, y1, x2, y2 = detection['box']
        position = detection['position']
        
        # Color coding
        if class_name.isupper():  # White pieces
            color = (255, 255, 255)
        elif class_name.islower():  # Black pieces  
            color = (0, 0, 0)
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Position label
        label = f"{class_name}@{position}"
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw moves panel
    frame_height, frame_width = frame.shape[:2]
    panel_width = 300
    panel_height = 200
    panel_x = frame_width - panel_width - 10
    panel_y = 10
    
    # Panel background
    cv2.rectangle(annotated_frame, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    
    # Title
    cv2.putText(annotated_frame, "RECENT MOVES", 
                (panel_x + 10, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Recent moves
    y_offset = 50
    for i, move_info in enumerate(recent_moves[-5:]):  # Last 5 moves
        move_text = f"{i+1}. {move_info['move']}"
        cv2.putText(annotated_frame, move_text, 
                    (panel_x + 10, panel_y + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
    
    return annotated_frame

def save_game_data(moves_history, output_path):
    """Save game data to JSON file"""
    # Convert numpy float32 to regular Python float
    serializable_moves = []
    for move in moves_history:
        serializable_move = {
            'frame': int(move['frame']),
            'time': float(move['time']),
            'move': str(move['move']),
            'piece': str(move['piece']),
            'from': str(move['from']),
            'to': str(move['to']),
            'confidence': float(move['confidence']),
            'distance': float(move['distance'])
        }
        serializable_moves.append(serializable_move)
    
    game_data = {
        'moves': serializable_moves,
        'total_moves': len(moves_history),
        'parameters': {
            'min_confidence': 0.75,
            'min_distance': 0.4,
            'chess_rules': True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(game_data, f, indent=2, ensure_ascii=False)

def analyze_chess_video(video_path, model_path="model_dataset/PieceDetection/best.pt"):
    """Main function to analyze chess video"""
    
    print(f"♟️ CHESS VIDEO ANALYZER")
    print("=" * 60)
    print(f"📹 Video: {os.path.basename(video_path)}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video info: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
    
    # Setup output
    output_path = f"analyzed_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing variables
    frame_idx = 0
    prev_positions = {}
    moves_history = []
    recent_moves = deque(maxlen=5)
    
    print(f"\n🔄 Processing video...")
    print("Controls: 'q'=quit, 's'=save frame, SPACE=pause")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_time = frame_idx / fps
            
            # Detect pieces
            curr_positions, detection_details = detect_pieces_in_frame(model, frame)
            
            # Detect moves
            if frame_idx > 1 and prev_positions:
                detected_moves = detect_moves(prev_positions, curr_positions)
                
                for move in detected_moves:
                    move_info = {
                        'frame': frame_idx,
                        'time': current_time,
                        **move
                    }
                    moves_history.append(move_info)
                    recent_moves.append(move_info)
                    
                    print(f"🎯 MOVE: {move['move']} at {current_time:.1f}s (conf: {move['confidence']:.2f})")
            
            # Annotate frame
            annotated_frame = annotate_frame_with_moves(frame, detection_details, list(recent_moves))
            
            # Add frame info
            info_text = f"Frame: {frame_idx} | Time: {current_time:.1f}s | Moves: {len(moves_history)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display and save
            cv2.imshow('Chess Video Analyzer', annotated_frame)
            out.write(annotated_frame)
            
            # Progress
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"📈 Progress: {progress:.1f}% | Moves: {len(moves_history)}")
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
            elif key == ord('s'):
                cv2.imwrite(f"frame_{frame_idx:06d}.jpg", annotated_frame)
                print(f"💾 Saved frame {frame_idx}")
            
            prev_positions = curr_positions.copy()
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Save game data
    if moves_history:
        json_path = f"game_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.json"
        save_game_data(moves_history, json_path)
        print(f"📄 Game data saved: {json_path}")
    
    # Summary
    print(f"\n📈 ANALYSIS COMPLETE:")
    print(f"🎞️ Processed {frame_idx} frames")
    print(f"♟️ Detected {len(moves_history)} moves")
    print(f"💾 Output: {output_path}")
    
    if moves_history:
        print(f"\n🎯 DETECTED MOVES:")
        for i, move in enumerate(moves_history):
            print(f"{i+1:2d}. {move['move']:<12} at {move['time']:6.1f}s")
    
    return True

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = select_video_file()
        if not video_path:
            print("❌ No video selected")
            return
    
    analyze_chess_video(video_path)

if __name__ == "__main__":
    main()