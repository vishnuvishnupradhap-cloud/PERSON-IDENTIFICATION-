import cv2
import numpy as np

class ColorClassifier:
    def __init__(self):
        """
        Initializes the HSV-based dress color classifier.
        """
        pass

    def get_dominant_color(self, hsv_crop):
        """
        Finds the dominant color name in an HSV cropped image using vectorized classification.
        :param hsv_crop: HSV cropped image.
        :return: string name of the dominant color.
        """
        if hsv_crop.size == 0:
            return "Unknown"

        h = hsv_crop[:, :, 0]
        s = hsv_crop[:, :, 1]
        v = hsv_crop[:, :, 2]
        
        # 1. Black: low brightness
        is_black = v < 55
        
        # 2. White: low saturation, high brightness
        is_white = (~is_black) & (s < 35) & (v > 185)
        
        # 3. Grey: low saturation, medium brightness
        is_grey = (~is_black) & (~is_white) & (s < 45) & (v >= 55) & (v <= 185)
        
        # Remaining pixels are considered colored
        colored = (~is_black) & (~is_white) & (~is_grey)
        
        # 4. Brown: dark orange/yellow
        # Brown is typically Hue [5, 22], Saturation >= 40, Value [40, 150]
        is_brown = colored & (((h >= 4) & (h <= 21)) & (s >= 40) & (v >= 40) & (v <= 150))
        
        # Keep rest for other colors
        remaining = colored & (~is_brown)
        
        # 5. Red: Hue [0, 8] or [168, 180]
        is_red = remaining & ((h <= 8) | (h > 168))
        
        # 6. Orange: Hue [9, 22]
        is_orange = remaining & ((h > 8) & (h <= 22))
        
        # 7. Yellow: Hue [23, 38]
        is_yellow = remaining & ((h > 22) & (h <= 38))
        
        # 8. Green: Hue [39, 85]
        is_green = remaining & ((h > 38) & (h <= 85))
        
        # 9. Blue: Hue [86, 132]
        is_blue = remaining & ((h > 85) & (h <= 132))
        
        # 10. Purple: Hue [133, 152]
        is_purple = remaining & ((h > 132) & (h <= 152))
        
        # 11. Pink: Hue [153, 168]
        is_pink = remaining & ((h > 152) & (h <= 168))
        
        # Count pixels for each color class
        counts = {
            "Black": np.sum(is_black),
            "White": np.sum(is_white),
            "Grey": np.sum(is_grey),
            "Brown": np.sum(is_brown),
            "Red": np.sum(is_red),
            "Orange": np.sum(is_orange),
            "Yellow": np.sum(is_yellow),
            "Green": np.sum(is_green),
            "Blue": np.sum(is_blue),
            "Purple": np.sum(is_purple),
            "Pink": np.sum(is_pink)
        }
        
        # Return the color with the highest pixel count
        dominant_color = max(counts, key=counts.get)
        
        # If the dominant color has 0 pixels (e.g. empty mask due to some anomaly), default to Grey
        if counts[dominant_color] == 0:
            return "Grey"
            
        return dominant_color

    def get_clothing_colors(self, frame, bbox):
        """
        Crops upper and lower body parts, and classifies their dominant colors.
        :param frame: full video frame in BGR.
        :param bbox: bounding box in format [left, top, w, h].
        :return: dict with keys "upper" and "lower" representing color names.
        """
        left, top, w, h = bbox
        h_img, w_img = frame.shape[:2]
        
        # Bound coordinates
        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = min(w_img, int(left + w))
        y2 = min(h_img, int(top + h))
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        if box_width <= 0 or box_height <= 0:
            return {"upper": "Unknown", "lower": "Unknown"}
            
        # Crop the full person bounding box
        person_crop = frame[y1:y2, x1:x2]
        
        # Heuristic crop zones for clothing:
        # Upper body (chest/torso): 15% to 45% height, 20% to 80% width
        uy1 = int(box_height * 0.15)
        uy2 = int(box_height * 0.45)
        ux1 = int(box_width * 0.20)
        ux2 = int(box_width * 0.80)
        
        # Lower body (legs/pants): 50% to 85% height, 20% to 80% width
        ly1 = int(box_height * 0.50)
        ly2 = int(box_height * 0.85)
        lx1 = int(box_width * 0.20)
        lx2 = int(box_width * 0.80)
        
        # Crop upper body and lower body regions
        upper_crop = person_crop[uy1:uy2, ux1:ux2]
        lower_crop = person_crop[ly1:ly2, lx1:lx2]
        
        # Convert crops to HSV
        upper_hsv = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2HSV) if upper_crop.size > 0 else np.empty((0,0,3), dtype=np.uint8)
        lower_hsv = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2HSV) if lower_crop.size > 0 else np.empty((0,0,3), dtype=np.uint8)
        
        # Classify dominant color
        upper_color = self.get_dominant_color(upper_hsv)
        lower_color = self.get_dominant_color(lower_hsv)
        
        return {
            "upper": upper_color,
            "lower": lower_color
        }
