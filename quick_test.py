#!/usr/bin/env python3
"""
Quick test to verify project functionality after cleanup
"""
import cv2
import os
from ultralytics import YOLO
import sys

def test_image_detection(image_path, model_path="model_dataset/PieceDetection/best.pt"):
    """Quick test of image detection"""
    print(f"🧪 QUICK PROJECT TEST")
    print("=" * 50)
    print(f"📷 Testing image: {os.path.basename(image_path)}")
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Load model
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return False
        
        print(f"📏 Image size: {img.shape}")
        
        # Run detection
        results = model.predict(img, verbose=False, conf=0.4)
        
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                print(f"🎯 Found {len(detections)} detections")
                
                # Count pieces
                class_names = model.names
                piece_counts = {}
                
                for conf, cls in zip(detections.conf.cpu().numpy(), detections.cls.cpu().numpy()):
                    class_name = class_names[int(cls)] if int(cls) in class_names else 'unknown'
                    if conf > 0.7:  # High confidence only
                        piece_counts[class_name] = piece_counts.get(class_name, 0) + 1
                
                print(f"\n📊 High confidence detections:")
                for piece, count in sorted(piece_counts.items()):
                    if piece not in ['board', 'chessboards-and-pieces']:
                        print(f"   {piece}: {count}")
                
                # Save result
                result_img = results[0].plot()
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                output_path = f"quick_test_result.jpg"
                cv2.imwrite(output_path, result_img_bgr)
                print(f"\n💾 Result saved: {output_path}")
                
                total_pieces = sum(count for piece, count in piece_counts.items() 
                                 if piece not in ['board', 'chessboards-and-pieces'])
                print(f"✅ Project working! Detected {total_pieces} chess pieces")
                return True
            else:
                print(f"⚪ No pieces detected")
        else:
            print(f"❌ No detection results")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return False

def main():
    print("🚀 TESTING PROJECT AFTER CLEANUP")
    print("=" * 60)
    
    # Test with available images
    test_images = []
    
    # Check for test images in yolo_dataset
    test_dir = "yolo_dataset/test/images"
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            test_images.append(os.path.join(test_dir, images[0]))
    
    # Check for any PNG/JPG in current directory
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        files = glob.glob(ext)
        if files:
            test_images.extend(files[:1])  # Add first file of each type
    
    if not test_images:
        print("❌ No test images found")
        print("💡 Please provide an image path as argument:")
        print("   python quick_test.py path/to/chess_image.jpg")
        return
    
    # Use command line arg if provided
    if len(sys.argv) > 1:
        test_images = [sys.argv[1]]
    
    # Test with first available image
    success = test_image_detection(test_images[0])
    
    if success:
        print(f"\n🎉 PROJECT TEST SUCCESSFUL!")
        print(f"✅ Model detection: Working")
        print(f"✅ Image processing: Working") 
        print(f"✅ File I/O: Working")
        print(f"\n🚀 Project ready for use!")
    else:
        print(f"\n⚠️ PROJECT TEST FAILED")
        print(f"🔧 Please check model and image paths")

if __name__ == "__main__":
    main()