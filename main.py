if __name__ == "__main__":
    # Create a database instance
    db = FaceFeaturesDB("face_features.db")
    
    # Check if dlib is installed and models can be loaded
    if db.has_dlib:
        print("Dlib is ready and models are loaded")
        
        # Example of adding a face from an image
        print("Example face recognition from webcam:")
        print("1. Press 'a' to add a new face (you will be asked to enter a name)")
        print("2. Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open webcam")
        else:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Cannot read frame from webcam")
                    break
                
                # Display original frame
                cv2.imshow("Webcam", frame)
                
                # Detect faces
                faces = db.extractor.detect_faces(frame)
                
                # Draw bounding boxes and labels
                for x, y, w, h in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display frame with bounding boxes
                cv2.imshow("Detected Faces", frame)
                
                key = cv2.waitKey(1)
                
                # Press 'a' to add a face
                if key == ord('a'):
                    # Detect faces
                    faces = db.extractor.detect_faces(frame)
                    
                    if len(faces) > 0:
                        name = input("Enter name for this face: ")
                        
                        # Add face to database
                        ids = db.add_face_from_image(name, frame)
                        print(f"Added {len(ids)} faces with name '{name}'")
                    else:
                        print("No faces detected")
                
                # Press 'q' to quit
                elif key == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Dlib is not available or models could not be loaded")
        
        # Use basic functions that don't depend on dlib
        # Example feature vector (128-dimensional vector commonly used in face recognition)
        example_vector = np.random.rand(128)
        
        # Add a face to the database
        face_id = db.add_face("John Doe", example_vector)
        print(f"Added face with ID: {face_id}")
        
        # Retrieve face by ID
        name, vector = db.get_feature_by_id(face_id)
        print(f"Retrieved face: {name}")
        print(f"Vector shape: {vector.shape}")