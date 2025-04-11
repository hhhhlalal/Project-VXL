
if __name__ == "__main__":
    # Create a database instance
    db = FaceFeatureDatabase("face_features.db")
    
    # Example feature vector (128-dimensional vector commonly used in face recognition)
    example_vector = np.random.rand(128)
    
    # Add a face to the database
    face_id = db.add_face("John Doe", example_vector)
    print(f"Added face with ID: {face_id}")
    
    # Retrieve the face by ID
    name, vector = db.get_feature_by_id(face_id)
    print(f"Retrieved face: {name}")
    print(f"Vector shape: {vector.shape}")
    
    # Get all faces
    all_faces = db.get_all_features()
    print(f"Total faces in database: {len(all_faces)}")
    
    # Get features by name
    john_features = db.get_features_by_name("John Doe")
    print(f"Features for John Doe: {len(john_features)}")
    
    # Get dictionary of features
    features_dict = db.get_feature_vectors_dict()
    for name, vectors in features_dict.items():
        print(f"{name}: {len(vectors)} feature vectors")