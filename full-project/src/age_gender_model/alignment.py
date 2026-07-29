import math
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    import torch
    import numpy as np
except ImportError:
    MTCNN = None

def detect_and_align_face(image: Image.Image, device: str = 'cpu') -> Image.Image:
    """
    Detects a face in the image and aligns it based on eye landmarks.
    Returns a cropped and aligned 224x224 PIL Image of the face.
    """
    if MTCNN is None:
        print("Warning: facenet_pytorch not installed. Returning resized original image.")
        return image.resize((224, 224))
        
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)
    
    # 1. Detect face and landmarks
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    
    if boxes is None or landmarks is None:
        print("Warning: No face detected. Using center crop/resize.")
        return image.resize((224, 224))
        
    # Landmarks: 0: left eye, 1: right eye
    left_eye = landmarks[0][0]
    right_eye = landmarks[0][1]
    
    # Calculate angle for alignment
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    
    # Rotate image around the eye center to make eyes horizontal
    aligned_full_image = image.rotate(angle, center=eye_center, resample=Image.BILINEAR)
    
    # Re-detect and crop the now strictly aligned face
    face_tensor = mtcnn(aligned_full_image)
    
    if face_tensor is None:
        print("Warning: Failed to crop aligned face. Using un-aligned crop.")
        face_tensor = mtcnn(image)
        
    if face_tensor is None:
        return image.resize((224, 224))
        
    # Convert tensor (values 0-255) back to PIL Image
    face_array = face_tensor.permute(1, 2, 0).byte().cpu().numpy()
    aligned_face = Image.fromarray(face_array)
    
    return aligned_face
