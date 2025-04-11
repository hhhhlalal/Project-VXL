import os
import cv2
import numpy as np
import dlib

class FaceExtractor:
    def __init__(self, model_path="../models/"):
        # Initialize face detector (HOG-based)
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load face landmark predictor
        predictor_path = os.path.join(model_path, "shape_predictor.dat")
        self.shape_predictor = dlib.shape_predictor(predictor_path)
    
    def detect_faces(self, image):
        # Convert to RGB if needed (dlib expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Detect faces
        faces = self.face_detector(rgb_image, 1)
        
        # Convert dlib results to list of rectangles (x, y, w, h)
        face_rects = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            face_rects.append((x, y, w, h))
            
        return face_rects
    
    def extract_face(self, image, face_rect, target_size=(150, 150)):
        x, y, w, h = face_rect
        
        # Add some margin
        margin_w = int(w * 0.1)
        margin_h = int(h * 0.1)
        
        # Calculate coordinates with margin
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # Extract face region
        face_img = image[y1:y2, x1:x2]
        
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        return face_img
    
    def get_landmarks(self, image, face_rect):
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
            
        # Convert rectangle to dlib format
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get landmarks
        shape = self.shape_predictor(rgb_image, dlib_rect)
        
        # Convert to numpy array
        landmarks = []
        for i in range(shape.num_parts):
            landmarks.append((shape.part(i).x, shape.part(i).y))
            
        return np.array(landmarks)
    
    def align_face(self, image, landmarks, target_size=(150, 150)):
        # Define reference points (eyes corners)
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate desired eye position
        eye_dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = target_size[0] * 0.3
        scale = desired_dist / eye_dist
        
        # Get rotation matrix
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Adjust translation
        tx = target_size[0] * 0.5 - center[0]
        ty = target_size[1] * 0.4 - center[1]  # Place eyes at 40% from top
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        
        # Apply transformation
        aligned_face = cv2.warpAffine(image, rotation_matrix, target_size)
        
        return aligned_face
    
    def process_image(self, image_path, output_dir=None, align=True):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect faces
        face_rects = self.detect_faces(image)
        
        faces = []
        for i, face_rect in enumerate(face_rects):
            if align:
                # Get landmarks and align face
                landmarks = self.get_landmarks(image, face_rect)
                face_img = self.align_face(image, landmarks)
            else:
                # Extract face without alignment
                face_img = self.extract_face(image, face_rect)
            
            faces.append(face_img)
            
            # Save face if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                face_name = f"{os.path.splitext(base_name)[0]}_face{i}.jpg"
                output_path = os.path.join(output_dir, face_name)
                cv2.imwrite(output_path, face_img)
        
        return faces