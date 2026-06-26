import argparse
import cv2
import time
import os
import sys

from pipeline import ReIDPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Person Detection, Tracking, and Re-identification Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="0",
        help="Camera device index (e.g. 0) or path to a video file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Re-ID cosine similarity threshold (default: 0.75)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--gallery",
        type=str,
        default="gallery.pkl",
        help="Path to save/load Re-ID gallery database"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run models on ('cuda', 'cpu'). Defaults to auto-detect."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse input source: check if it's an integer index (webcam)
    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)
        print(f"[Run] Using webcam device {input_source} as input.")
    else:
        if not os.path.exists(input_source):
            print(f"[Error] Video file not found: {input_source}")
            sys.exit(1)
        print(f"[Run] Using video file '{input_source}' as input.")
        
    # Initialize the tracking & Re-ID pipeline
    pipeline = ReIDPipeline(
        conf_threshold=args.conf,
        similarity_threshold=args.threshold,
        gallery_path=args.gallery,
        device=args.device
    )
    
    # Open video capture source
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"[Error] Failed to open video source: {args.input}")
        sys.exit(1)
        
    print("\n--- Pipeline Started Successfully ---")
    print("Controls:")
    print("  'q' - Quit and save database")
    print("  's' - Explicitly save gallery database")
    print("--------------------------------------\n")
    
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Run] End of video stream or failed to read frame.")
                break
                
            frame_count += 1
            
            # Process the frame
            start_time = time.time()
            annotated_frame, tracks_info = pipeline.process_frame(frame)
            proc_time = time.time() - start_time
            
            # Compute FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # Draw FPS and process time info on frame
            info_text = f"FPS: {fps:.1f} | Latency: {proc_time*1000:.1f}ms | People Tracked: {len(tracks_info)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Show output
            cv2.imshow("Person Re-ID Pipeline", annotated_frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Run] Quitting...")
                break
            elif key == ord('s'):
                pipeline.save_state()
                
    except KeyboardInterrupt:
        print("[Run] Interrupted by user.")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save gallery state on exit
        pipeline.save_state()
        print("[Run] Cleanup finished. Exiting.")

if __name__ == "__main__":
    main()
