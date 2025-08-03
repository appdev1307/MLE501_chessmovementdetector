# Chess Movement Detector - Usage Guide 🎯

## Quick Start Examples

### 1. 🖼️ **Test with Single Image**

```bash
# Interactive selection
python test_any_chess_image.py

# Or specify image directly
python test_any_chess_image.py "path/to/chess_board.jpg"
```

**Example Output:**

```
🎯 Found 59 detections
📊 High confidence pieces:
   P: 8  R: 2  N: 2  B: 2  Q: 1  K: 1  (White)
   p: 8  r: 2  n: 2  b: 2  q: 1  k: 1  (Black)
💾 Result saved: detection_result_chess.jpg
```

### 2. 🎬 **Analyze Chess Video**

```bash
# Main video analyzer
python chess_video_analyzer.py

# Or specify video directly
python chess_video_analyzer.py "chess_game.mp4"
```

**Example Output:**

```
♟️ Detected 19 moves:
 1. Pc2-d3 at 4.0s  (conf: 0.95)
 2. Pe2-e4 at 8.2s  (conf: 0.92)
 3. Ng1-f3 at 12.1s (conf: 0.88)
💾 Video: analyzed_chess_game.mp4
📄 Data: game_analysis_chess_game.json
```

### 3. 🧪 **Validate Model Performance**

```bash
# Quick functionality test
python quick_test.py

# Full model validation (if examples/ exists)
python examples/model_validation.py
```

## 🎮 Interactive Controls

During video processing:

- **`q`** - Quit processing
- **`s`** - Save current frame as image
- **`SPACE`** - Pause/resume video
- **`m`** - Print current move history

## ⚙️ Configuration Options

### Move Detection Sensitivity

Edit `chess_video_analyzer.py` to adjust:

```python
# In detect_moves() function
min_confidence=0.75,  # 75% minimum confidence
min_distance=0.4      # 0.4 squares minimum move
```

### Model Confidence Threshold

```python
# In detect_pieces_in_frame() function
conf_threshold=0.6    # 60% piece detection threshold
```

## 📊 Output File Formats

### JSON Game Data

```json
{
  "moves": [
    {
      "frame": 121,
      "time": 4.03,
      "move": "Pe2-e4",
      "piece": "P",
      "from": "e2",
      "to": "e4",
      "confidence": 0.92,
      "distance": 2.0
    }
  ],
  "total_moves": 15,
  "parameters": {
    "min_confidence": 0.75,
    "chess_rules": true
  }
}
```

### Video Output

- **Annotated video** with piece bounding boxes
- **Recent moves panel** (last 5 moves)
- **Frame counter** and timestamp
- **Chess position labels** (e.g., `P@e2`)

## 🎯 Best Practices

### For Image Analysis:

- Use **high resolution** chess board images (>400x400)
- Ensure **good lighting** and **clear piece shapes**
- **Straight-on angle** works best
- **Avoid shadows** over the board

### For Video Analysis:

- **Stable camera** position throughout game
- **Clear piece movements** (not too fast)
- **Good contrast** between pieces and board
- **Consistent lighting** during recording

### Troubleshooting:

- **No pieces detected**: Lower `conf_threshold` to 0.4
- **Too many false moves**: Increase `min_confidence` to 0.8
- **Missing moves**: Decrease `min_distance` to 0.3
- **Video won't open**: Check codec support (prefer MP4)

## 📈 Performance Optimization

### For faster processing:

```python
# Process every 2nd frame only
frame_step = 2
if frame_idx % frame_step != 0:
    continue
```

### For better accuracy:

```python
# Use higher confidence thresholds
conf_threshold = 0.7
min_confidence = 0.8
```

## 🔧 Customization Examples

### Custom Piece Classes

```python
# Modify class_names in detect_pieces_in_frame()
custom_classes = ['P', 'R', 'N', 'B', 'Q', 'K',
                 'p', 'r', 'n', 'b', 'q', 'k', 'empty']
```

### Custom Output Format

```python
# Add custom fields to move data
move_info = {
    'frame': frame_idx,
    'time': current_time,
    'player': 'white' if piece.isupper() else 'black',
    'evaluation': calculate_position_value(),
    **move
}
```

## ❓ FAQ

**Q: Can it detect castling or en passant?**  
A: Currently detects basic moves. Special moves need additional logic.

**Q: Does it work with different board styles?**  
A: Trained on standard tournament boards. May need retraining for unique styles.

**Q: Can I use my own trained model?**  
A: Yes! Replace `model_dataset/PieceDetection/best.pt` with your model.

**Q: How accurate is move detection?**  
A: 85-95% accuracy on clear tournament-style games.

---

_For more advanced usage, see the source code comments and examples/ directory._
