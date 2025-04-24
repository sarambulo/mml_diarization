import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm

def extract_lip_coordinates(bbox, predictor):
    height, width = bbox.shape[:2]
    face_rect = dlib.rectangle(0, 0, width, height)
    landmarks = predictor(bbox, face_rect)
    landmarks = face_utils.shape_to_np(landmarks)
    
    # Get mouth region (indices 48-68)
    mouth_indices = list(range(48, 68))
    mouth_landmarks = landmarks[mouth_indices]
    #bounding box
    x_min = np.min(mouth_landmarks[:, 0])
    y_min = np.min(mouth_landmarks[:, 1])
    x_max = np.max(mouth_landmarks[:, 0])
    y_max = np.max(mouth_landmarks[:, 1])
    #add smol padding
    padding = int(0.1 * (x_max - x_min))
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)


def preprocess_face_batch(face_batch):
    # Normalize the entire batch to [0, 1] range
    if face_batch.min() < 0 or face_batch.max() > 1:
        face_batch = (face_batch - face_batch.min()) / (face_batch.max() - face_batch.min())
    
    # Convert to uint8 (0-255 range)
    face_batch = (face_batch * 255).astype(np.uint8)
    
    # Transpose from [batch_size, C, H, W] to [batch_size, H, W, C]
    # This preserves the batch dimension while changing the layout of each image
    face_batch = np.transpose(face_batch, (0, 2, 3, 1))
    
    return face_batch

def extract_and_crop_lips(face_batch, predictor, target_size=(64, 32)):
    lips_batch = []
    
    for face in face_batch:
        x_min, y_min, x_max, y_max = extract_lip_coordinates(face, predictor)
        lip = face[y_min:y_max, x_min:x_max]
        lip_resized = cv2.resize(lip, target_size, interpolation=cv2.INTER_CUBIC)
        lips_batch.append(lip_resized)
        
    lips_batch = np.array(lips_batch)
    return lips_batch

def main():
    data_path = '../preprocessed/00001/Chunk_1/face_0.npy'
    predictor_path = "shape_predictor_68_face_landmarks_GTX.dat"
    predictor = dlib.shape_predictor(predictor_path)
    face_batch = np.load(data_path)
    face_batch_processed = preprocess_face_batch(face_batch)
    
    lip_target_size = (64, 32)
    
    lips_batch = extract_and_crop_lips(face_batch_processed, predictor, lip_target_size)
    
    # Save standardized lip batch
    np.save("lip_0.npy", lips_batch)

if __name__ == "__main__":
    main()