import pandas as pd
import os
from motion_classifier_dataset import MotionClassifierDataset

def check_class_distribution():
    # Load the dataframe exactly as done in training
    csv_path = os.path.abspath("motions.csv")
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Instantiate the dataset
    dataset = MotionClassifierDataset(df)
    
    # Get the class names directly from your dataset class
    class_names = MotionClassifierDataset.MOTION_CLASSES
    
    # Initialize counters
    class_counts = {name: 0 for name in class_names}
    
    print(f"Total samples in dataset: {len(dataset)}")
    print("Counting classes (this might take a moment)...\n")
    
    # Iterate through the dataset
    for i in range(len(dataset)):
        # The dataset __getitem__ returns: conf1, conf2, residues, motion_class, lengths
        label = dataset[i][3] 
        class_name = class_names[label]
        class_counts[class_name] += 1
        
    # Display the results
    print("=== Dataset Class Makeup ===")
    total = sum(class_counts.values())
    
    for name, count in class_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{name}: {count} samples ({percentage:.2f}%)")

if __name__ == "__main__":
    check_class_distribution()
