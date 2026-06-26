import torch
import cv2
import numpy as np
from torchreid.reid.utils import FeatureExtractor

class PersonEmbedder:
    def __init__(self, model_name="osnet_x1_0", device=None):
        """
        Initializes the Torchreid Re-ID feature extractor.
        :param model_name: OSNet model variant (e.g., 'osnet_x1_0', 'osnet_x0_75').
        :param device: 'cuda' or 'cpu'. If None, automatically detects.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Embedder] Initializing Torchreid Feature Extractor ({model_name}) on device: {self.device}")
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=self.device
        )
        
    def extract(self, frame, bboxes):
        """
        Extracts embedding vectors for all bounding boxes in the frame.
        :param frame: full video frame (BGR numpy array).
        :param bboxes: list of bounding boxes in format [left, top, w, h].
        :return: numpy array of shape (num_bboxes, 512) containing Re-ID features.
        """
        if not bboxes:
            return np.empty((0, 512), dtype=np.float32)
            
        crops = []
        h_img, w_img = frame.shape[:2]
        
        for bbox in bboxes:
            left, top, w, h = bbox
            # Ensure coordinates are within image boundaries
            x1 = max(0, int(left))
            y1 = max(0, int(top))
            x2 = min(w_img, int(left + w))
            y2 = min(h_img, int(top + h))
            
            # Crop the bounding box
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                # If the crop is empty, use a blank placeholder image to avoid crashing
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
            crops.append(crop)
            
        # Extract features using torchreid
        # extractor returns a PyTorch tensor on CPU or GPU
        features_tensor = self.extractor(crops)
        
        # Convert to numpy array on CPU
        features_numpy = features_tensor.cpu().numpy()
        
        # L2 Normalize the features so that cosine similarity is simply the dot product
        norms = np.linalg.norm(features_numpy, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-12, norms)
        features_normalized = features_numpy / norms
        
        return features_normalized
