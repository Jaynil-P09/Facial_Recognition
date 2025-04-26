import os
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# Directory structure: data/Person1/img1.jpg, data/Person2/img2.jpg, ...
data_dir = "data"
model_name = "ArcFace"

known_faces = {}

# Loop through each person in the dataset
for person_name in tqdm(os.listdir(data_dir), desc="Processing People"):
    person_path = os.path.join(data_dir, person_name)
    if os.path.isdir(person_path):
        embeddings = []
        for img_file in os.listdir(person_path):
            # Filter out non-image files
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(person_path, img_file)
            try:
                result = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)
                embedding = result[0]['embedding']
                embeddings.append(embedding)
                print(f"✓ Processed {img_path}")
            except Exception as e:
                print(f"✗ Failed to process {img_path}: {e}")

        if embeddings:
            # Option 1: store all embeddings
            known_faces[person_name] = np.array(embeddings)

            # Option 2: store only the average embedding
            # known_faces[person_name] = np.mean(embeddings, axis=0)

# Save to .npy file
np.save("known_faces.npy", known_faces)
print("\n✅ All known face embeddings saved to 'known_faces.npy'")







