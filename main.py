import os
import sys
import cv2
import numpy as np
import json
import argparse
from face_extractor import FaceExtractor
from feature_extractor import FeatureExtractor
from matching_machine import FaceMatcher
from database import FaceDatabase

def load_config(config_path="../config/settings.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default config
        return {
            "db_path": "face_features.db",
            "known_faces_dir": "../data/known_faces",
            "unknown_faces_dir": "../data/unknown_faces",
            "models_dir": "../models",
            "matching_threshold": 0.6
        }

def process_known_faces(config):
    """Process known faces and add to database"""
    print("Processing known faces...")
    
    # Initialize components
    face_extractor = FaceExtractor(model_path=config["models_dir"])
    feature_extractor = FeatureExtractor(model_path=config["models_dir"])
    db = FaceDatabase(db_path=config["db_path"])
    
    # Process each person's directory
    known_faces_dir = config["known_faces_dir"]
    total_processed = 0
    
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        print(f"Processing person: {person_name}")
        person_faces = 0
        
        # Process each image
        for image_file in os.listdir(person_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, image_file)
                
                try:
                    # Extract features
                    results = feature_extractor.extract_features_from_image(image_path)
                    
                    # Add to database
                    for _, feature_vector in results:
                        db.add_face(person_name, feature_vector, image_path)
                        person_faces += 1
                        total_processed += 1
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        print(f"  Added {person_faces} faces for {person_name}")
    
    print(f"Total faces processed: {total_processed}")
    return total_processed

def recognize_unknown_faces(config):
    """Recognize faces in the unknown_faces directory"""
    print("Recognizing unknown faces...")
    
    # Initialize components
    feature_extractor = FeatureExtractor(model_path=config["models_dir"])
    db = FaceDatabase(db_path=config["db_path"])
    
    # Create face matcher
    matcher = FaceMatcher(
        feature_extractor=feature_extractor,
        threshold=config["matching_threshold"]
    )
    
    # Load known faces from database
    all_faces = db.get_all_faces()
    for face_id, name, feature_vector, _ in all_faces:
        matcher.add_known_face(name, feature_vector)
    
    print(f"Loaded {len(all_faces)} known faces")
    
    # Process unknown faces
    unknown_dir = config["unknown_faces_dir"]
    results = []
    
    for image_file in os.listdir(unknown_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(unknown_dir, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not load image: {image_path}")
                    continue
                
                # Recognize faces
                recognition_results = matcher.recognize_faces_in_image(image_path)
                
                # Draw results on image
                result_image = image.copy()
                
                for face_rect, name, confidence in recognition_results:
                    x, y, w, h = face_rect
                    
                    # Draw rectangle
                    if name == "Unknown":
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        color = (0, 255, 0)  # Green for known
                    
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"{name} ({confidence:.2f})"
                    cv2.putText(
                        result_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
                
                # Save result image
                base_name = os.path.basename(image_path)
                result_path = os.path.join(unknown_dir, "result_" + base_name)
                cv2.imwrite(result_path, result_image)
                
                print(f"Processed {image_path} - Found {len(recognition_results)} faces")
                results.append({
                    "image": image_path,
                    "result_image": result_path,
                    "faces": [
                        {
                            "name": name,
                            "confidence": float(confidence),
                            "position": face_rect
                        }
                        {
                            "name": name,
                            "confidence": float(confidence),
                            "position": face_rect
                        }
                        for face_rect, name, confidence in recognition_results
                    ]
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return results

def main():
    """Main function to run face recognition system"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--config", default="../config/settings.json", help="Path to configuration file")
    parser.add_argument("--mode", choices=["train", "recognize", "both"], default="both", 
                        help="Mode of operation: train, recognize, or both")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process according to mode
    if args.mode in ["train", "both"]:
        processed = process_known_faces(config)
        print(f"Training complete. Processed {processed} faces.")
    
    if args.mode in ["recognize", "both"]:
        results = recognize_unknown_faces(config)
        print(f"Recognition complete. Processed {len(results)} images.")
    
    print("Done.")

if __name__ == "__main__":
    main()