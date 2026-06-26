from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5):
        """
        Initializes the YOLO detector.
        :param model_name: YOLO model file name.
        :param conf_threshold: Confidence threshold for person detections.
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        # Class 0 in COCO is 'person'
        self.person_class_id = 0

    def detect(self, frame):
        """
        Detects people in the frame.
        :param frame: input image (numpy array, BGR).
        :return: list of detections in the format [[left, top, w, h], confidence, class_name]
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        # Results is a list of results (one per image/frame passed)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class label
                class_id = int(box.cls[0].item())
                if class_id != self.person_class_id:
                    continue
                
                # Get confidence
                conf = float(box.conf[0].item())
                if conf < self.conf_threshold:
                    continue
                
                # Get coordinates in xyxy format
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                
                # Convert to [left, top, width, height] format for DeepSORT
                left = int(x1)
                top = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                detections.append(([left, top, w, h], conf, "person"))
                
        return detections
