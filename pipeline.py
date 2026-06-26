import cv2
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detector import PersonDetector
from embedder import PersonEmbedder
from color_classifier import ColorClassifier
from gallery import ReIDGallery

class ReIDPipeline:
    def __init__(self, conf_threshold=0.5, similarity_threshold=0.75, gallery_path="gallery.pkl", device=None):
        """
        Initializes the entire Person Detection, Tracking, and Re-identification Pipeline.
        """
        self.detector = PersonDetector(conf_threshold=conf_threshold)
        self.embedder = PersonEmbedder(device=device)
        self.color_classifier = ColorClassifier()
        self.gallery = ReIDGallery(similarity_threshold=similarity_threshold)
        
        # Load existing gallery database if available
        self.gallery_path = gallery_path
        self.gallery.load_gallery(self.gallery_path)
        
        # Tracker initialization (custom embedder=None as we provide our own OSNet embeddings)
        self.tracker = DeepSort(
            max_age=30,          # frames to keep track alive after lost
            n_init=3,            # frames before track is confirmed
            nms_max_overlap=1.0, # disable tracker-side NMS as YOLO handles it
            embedder=None
        )
        
        # Mappings to persist track information across frames
        self.track_to_global_id = {}
        self.track_to_colors = {}

    def compute_iou(self, boxA, boxB):
        """
        Computes Intersection-over-Union (IoU) of two boxes in [left, top, right, bottom] format.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0
            
        return interArea / unionArea

    def process_frame(self, frame):
        """
        Processes a single video frame.
        :param frame: input frame (BGR numpy array).
        :return: processed frame with drawings, and list of current tracked person info dicts.
        """
        annotated_frame = frame.copy()
        
        # 1. Detect people
        detections = self.detector.detect(frame)
        
        # Prepare data structures
        bboxes_xywh = [d[0] for d in detections]
        
        # 2. Extract Re-ID embeddings (OSNet)
        embeddings = self.embedder.extract(frame, bboxes_xywh)
        
        # 3. Classify clothing colors
        colors_list = [self.color_classifier.get_clothing_colors(frame, bbox) for bbox in bboxes_xywh]
        
        # 4. Prepare detections format for DeepSORT
        # format: ( [left, top, w, h], confidence, detection_class )
        detections_for_tracker = []
        for i, det in enumerate(detections):
            bbox, conf, cls_name = det
            detections_for_tracker.append((bbox, conf, cls_name))
            
        # 5. Update the tracker with the custom embeddings
        tracks = self.tracker.update_tracks(detections_for_tracker, embeds=embeddings)
        
        # We will collect active track information for return
        active_tracks_info = []
        
        # 6. Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # Current Kalman filtered bounding box [left, top, right, bottom]
            orig_ltrb = track.to_ltrb(orig=True) # Original detection bounding box that matched this track
            
            global_id = None
            similarity = 0.0
            colors = {"upper": "Unknown", "lower": "Unknown"}
            
            # If the track was matched to a detection in the current frame
            if orig_ltrb is not None:
                # Find which original detection corresponds to this track (highest IoU)
                best_iou = 0.0
                best_det_idx = -1
                
                # orig_ltrb is [left, top, right, bottom]
                for idx, (bbox_xywh, _, _) in enumerate(detections):
                    l, t, w, h = bbox_xywh
                    det_ltrb = [l, t, l + w, t + h]
                    
                    iou = self.compute_iou(orig_ltrb, det_ltrb)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx
                        
                # If we found a matching detection with reasonable IoU
                if best_det_idx != -1 and best_iou > 0.4:
                    embedding = embeddings[best_det_idx]
                    colors = colors_list[best_det_idx]
                    
                    # Crop thumbnail for Re-ID gallery representation
                    l, t, r, b = [int(coord) for coord in orig_ltrb]
                    h_img, w_img = frame.shape[:2]
                    l = max(0, l)
                    t = max(0, t)
                    r = min(w_img, r)
                    b = min(h_img, b)
                    thumbnail = frame[t:b, l:r] if (r - l > 0 and b - t > 0) else None
                    
                    # Match this embedding against the global gallery
                    global_id, similarity, is_new = self.gallery.match_or_create(embedding, colors, thumbnail)
                    
                    # Update local caches
                    self.track_to_global_id[track_id] = global_id
                    self.track_to_colors[track_id] = colors
            
            # If the track wasn't matched to a detection (or detector missed), use cached info
            if global_id is None:
                global_id = self.track_to_global_id.get(track_id, "Pending")
                colors = self.track_to_colors.get(track_id, {"upper": "Unknown", "lower": "Unknown"})
                similarity = 0.0
                
            # Collect track data
            active_tracks_info.append({
                "global_id": global_id,
                "track_id": track_id,
                "bbox": ltrb,
                "colors": colors,
                "similarity": similarity
            })
            
            # 7. Draw Visualizations
            # Choose color: green for matched/old, cyan for new/pending
            box_color = (0, 255, 255) if global_id == "Pending" else (0, 255, 0)
            
            # Draw bounding box
            l, t, r, b = [int(c) for c in ltrb]
            cv2.rectangle(annotated_frame, (l, t), (r, b), box_color, 2)
            
            # Draw a sleek label background card
            label_text = f"{global_id} (Track: {track_id})"
            if similarity > 0:
                label_text += f" Match: {similarity:.2f}"
            
            color_text = f"U: {colors['upper']} | L: {colors['lower']}"
            
            # Position labels on top of the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            # Get text sizes
            (w1, h1), baseline1 = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            (w2, h2), baseline2 = cv2.getTextSize(color_text, font, font_scale, font_thickness)
            
            card_w = max(w1, w2) + 10
            card_h = h1 + h2 + 12
            
            # Draw semi-transparent card background
            card_top = max(0, t - card_h)
            card_bottom = card_top + card_h
            card_right = min(annotated_frame.shape[1], l + card_w)
            
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (l, card_top), (card_right, card_bottom), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # Draw text lines
            cv2.putText(annotated_frame, label_text, (l + 5, card_top + h1 + 4), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(annotated_frame, color_text, (l + 5, card_top + h1 + h2 + 8), font, font_scale, (255, 200, 100), font_thickness, cv2.LINE_AA)

        # Remove stale track IDs from local cache
        active_tracker_ids = {t.track_id for t in tracks}
        stale_ids = [tid for tid in list(self.track_to_global_id.keys()) if tid not in active_tracker_ids]
        for tid in stale_ids:
            self.track_to_global_id.pop(tid, None)
            self.track_to_colors.pop(tid, None)
            
        return annotated_frame, active_tracks_info

    def save_state(self):
        """
        Saves the global database to disk.
        """
        self.gallery.save_gallery(self.gallery_path)
