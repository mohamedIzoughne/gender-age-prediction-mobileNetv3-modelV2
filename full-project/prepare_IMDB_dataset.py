import cv2
import pandas as pd
import os

df = pd.read_csv('imdb_train_new_1024.csv')

output_dir = 'train_cropped_faces'

def extract_crop(row, idx):
    img_path = row['filename']

    print(img_path)
    basename = os.path.basename(img_path)
    img = cv2.imread(f'files/{basename}')
    
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return

    # Extract coordinates
    x_min, y_min = int(row['x_min']), int(row['y_min'])
    x_max, y_max = int(row['x_max']), int(row['y_max'])
    
    # Ensure coordinates are within image boundaries
    h, w, _ = img.shape
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    # Crop face
    face_crop = img[y_min:y_max, x_min:x_max]

    cv2.imwrite(os.path.join(output_dir, f"crop_{idx}.jpg"), face_crop)

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over DataFrame and extract faces
extract_crop(df.iloc[0, :], 0)
