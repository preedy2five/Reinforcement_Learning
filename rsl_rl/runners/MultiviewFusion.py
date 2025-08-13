import pickle
import pandas as pd


def load_embedding(file_path):
    all_camera_data = []

    try:
        with open(file_path, 'rb') as f:
            camera_data = pickle.load(f)
            all_camera_data.append(camera_data)
        print(f"Loaded {file_path} successfully.")
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    
    return all_camera_data

def concatenate_embeddings(all_camera_data,it,frameidx):
    

    feature_columns = all_camera_data.columns[2:]

    # Create a single row where each feature column contains a list of all values
    concatenated_row = {'timestamp': all_camera_data['timestamp'][0]}  # Use first timestamp or modify as needed
    
    for feature in feature_columns:
        concatenated_row[feature] = all_camera_data[feature].tolist()

    concatenated_df = pd.DataFrame([concatenated_row])
    # Print f2000 column 
    
    # Save to a .pkl file
    concatenated_df.to_pickle(f"Fusion_Out/{it}-{frameidx}")
    print("Concatenated embeddings saved to concatenated_frame_embeddings.pkl")

def frame_concatenation(file_path,it,frameidx):
    all_camera_data = load_embedding(file_path)
    concatenate_embeddings(all_camera_data,it,frameidx)


if __name__ == "__main__":
    file_path = "image_file_embeddings.pkl"
    frame_concatenation(file_path)