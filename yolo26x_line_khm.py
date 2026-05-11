
import os
#!pip install ultralytics --quiet
from ultralytics import YOLO
#!pip install datasets --quiet
from datasets import load_dataset
import cv2
import json
import glob
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET
import shutil
import random
import matplotlib.pyplot as plt
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
token = os.environ.get('HF_TOKEN')
login(token = token, add_to_git_credential  = True)

def make_yolo_dir(cwd):
    yolo_dir = os.path.join(cwd, 'DATA-YOLO')
    os.mkdir(yolo_dir)
    for i in ['train', 'validation', 'test']:
        folder = os.path.join(yolo_dir, i)
        os.mkdir(folder)
        for j in ['images', 'labels']:
            subfolder = os.path.join(folder, j)
            os.mkdir(subfolder)
    return yolo_dir


def get_label_mapping():
    """Define label to class ID mapping for layout elements"""
    labels = {
        'line': 0,

    }
    return labels

def polygon_to_yolo_bbox(points, img_width, img_height):
    """
    Convert polygon points to YOLO format bounding box
    YOLO format: <class_id> <x_center> <y_center> <width> <height> (all normalized)
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    # Convert to YOLO format (normalized)
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return x_center, y_center, width, height

def convert_json_to_yolo(json_path, label_mapping):
    """
    Convert a single JSON annotation file to YOLO format
    Returns list of YOLO format annotation strings
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image dimensions from the first shape or use default
    # We'll need to get actual dimensions from the image file
    img_path = data.get('imagePath', '')
    
    annotations = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower()
        points = shape.get('points', [])
        
        if label in label_mapping and len(points) >= 2:
            class_id = label_mapping[label]
            # We'll calculate bbox from polygon points
            # Note: We need actual image dimensions, will handle this in main function
            annotations.append({
                'class_id': class_id,
                'points': points,
                'label': label
            })
    
    return annotations, img_path

from datasets import concatenate_datasets
def prepare_combining_dataset_from_hf():
    """
    Prepare the COMBINING dataset for YOLO training from Hugging Face
    - Load dataset from rinabuoy/khmer-layout-data-standardized-with-line-annotations
    - Convert to YOLO format
    - Organize into train/val/test splits
    """
    print("Loading dataset from Hugging Face: rinabuoy/layout-data-khmer-1290-line-only")
    
    # Load dataset from Hugging Face
    '''
    dataset_v1 = load_dataset("rinabuoy/layout-data-khmer-1290-line-only")
    dataset_v2 = load_dataset("rinabuoy/line-data-khmer-dense-1")  
    dataset_v3 = load_dataset("rinabuoy/line-data-khmer-dense-2")
    dataset_v4 = load_dataset("rinabuoy/line-data-khmer-dense-3")  
    dataset_v5 = load_dataset("rinabuoy/line-data-khmer-dense-4")
    dataset_v6 = load_dataset("rinabuoy/line-data-khmer-aug-1")

    #dataset_v2 = load_dataset("rinabuoy/khmer-layout-data-2-line-annotations")
    #dataset['train'] = concatenate_datasets([dataset['train'], dataset_v2['train'], dataset_v3['train'], dataset_v4['train'], dataset_v5['train'], dataset_v6['train']])
    #dataset['validation'] = concatenate_datasets([dataset['validation'], dataset_v2['validation'], dataset_v3['validation'], dataset_v4['validation'], dataset_v5['validation'], dataset_v6['validation']])
    #dataset['test'] = concatenate_datasets([dataset['test'], dataset_v2['test'], dataset_v3['test'], dataset_v4['test'], dataset_v5['test'], dataset_v6['test']])
    #print(f"\nDataset loaded successfully!")
    #print(f"Available splits: {list(dataset.keys())}")
    
    dataset_v7 = load_dataset("rinabuoy/line-data-latin-aug-1")
    dataset_v8 = load_dataset("rinabuoy/line-data-latin-aug-2")
    dataset_v9 = load_dataset("rinabuoy/line-data-latin-aug-3")
    dataset_v10 = load_dataset("rinabuoy/line-data-latin-aug-4")
    #dataset_v11 = load_dataset("rinabuoy/line-data-id-card")
    #dataset['train'] = concatenate_datasets([dataset['train'], dataset_v2['train'], dataset_v3['train'], dataset_v4['train'], dataset_v5['train']])
    #dataset['validation'] = concatenate_datasets([dataset['validation'], dataset_v2['validation'], dataset_v3['validation'], dataset_v4['validation'], dataset_v5['validation']])
    #dataset['test'] = concatenate_datasets([dataset['test'], dataset_v2['test'], dataset_v3['test'], dataset_v4['test'], dataset_v5['test']])
    '''
    # Create YOLO dataset structure
    yolo_dataset_dir = os.path.join('/app/workspace', 'YOLO_DATASET_LINE')
    os.makedirs(yolo_dataset_dir, exist_ok=True)
    if True:
        return yolo_dataset_dir  # Skip processing if already done
    # Get label mapping
    label_mapping = get_label_mapping()

    all_datasets = [
        (dataset_v1,  "khmer-1290"),
        (dataset_v2,  "khmer-dense-1"),
        (dataset_v3,  "khmer-dense-2"),
        (dataset_v4,  "khmer-dense-3"),
        (dataset_v5,  "khmer-dense-4"),
        (dataset_v6,  "khmer-aug-1"),
        (dataset_v7,  "latin-aug-1"),
        (dataset_v8,  "latin-aug-2"),
        (dataset_v9,  "latin-aug-3"),
        (dataset_v10, "latin-aug-4"),
    ]

    # Process each split in the dataset
    for split_name in all_datasets[0][0].keys():
        print(f"\nProcessing {split_name} split...")

        # Create directories for this split
        os.makedirs(os.path.join(yolo_dataset_dir, split_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset_dir, split_name, 'labels'), exist_ok=True)

        successful = 0
        failed = 0
        skipped = 0

        for ds_idx, (dataset, ds_name) in enumerate(all_datasets):
            if split_name not in dataset:
                print(f"  {ds_name} has no '{split_name}' split, skipping.")
                continue
            split_data = dataset[split_name]

            for idx, example in enumerate(tqdm(split_data, desc=f"Processing {split_name} ({ds_name})")):
                try:
                    # Get image
                    image = example['image']  # PIL Image

                    # Parse annotation JSON string
                    annotation_data = json.loads(example['annotation'])
                    shapes = annotation_data.get('shapes', [])

                    # Get image dimensions
                    img_width, img_height = image.size

                    # Skip if no annotations
                    if len(shapes) == 0:
                        skipped += 1
                        continue

                    # Create YOLO OBB format annotations
                    yolo_annotations = []
                    for shape in shapes:
                        label = 'line'#shape.get('label', '').lower()
                        points = shape.get('points', [])

                        # Check if label is in our mapping
                        # FIXED: Changed from < 3 to < 2 to allow 2-point bounding boxes
                        if label not in label_mapping or len(points) < 2:
                            continue

                        class_id = label_mapping[label]

                        # Handle OBB format - need exactly 4 points for oriented bounding box
                        if len(points) < 4:
                            # If less than 4 points (e.g. 2 or 3), create a minimal bounding box
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            xmin, xmax = min(x_coords), max(x_coords)
                            ymin, ymax = min(y_coords), max(y_coords)
                            # Create 4 corners clockwise from top-left
                            obb_points = [
                                [xmin, ymin],  # top-left
                                [xmax, ymin],  # top-right
                                [xmax, ymax],  # bottom-right
                                [xmin, ymax]   # bottom-left
                            ]
                        elif len(points) == 4:
                            # Sort 4 points to ensure clockwise order from top-left
                            import numpy as np
                            pts = np.array(points, dtype=np.float32)
                            # Sort by y first (top to bottom), then by x (left to right)
                            sorted_pts = pts[np.argsort(pts[:, 1])]
                            # Top two points: sort by x (left to right)
                            top_points = sorted_pts[:2][np.argsort(sorted_pts[:2, 0])]
                            # Bottom two points: sort by x (right to left for clockwise)
                            bottom_points = sorted_pts[2:][np.argsort(sorted_pts[2:, 0])[::-1]]
                            # Arrange: top-left, top-right, bottom-right, bottom-left
                            obb_points = np.vstack([top_points, bottom_points]).tolist()
                        else:
                            # If more than 4 points, use minimum area rotated rectangle
                            import numpy as np
                            pts = np.array(points, dtype=np.float32)
                            rect = cv2.minAreaRect(pts)
                            box = cv2.boxPoints(rect)
                            # cv2.boxPoints returns points starting from bottom-left, counter-clockwise
                            # We need to reorder to: top-left, top-right, bottom-right, bottom-left (clockwise)
                            sorted_box = box[np.argsort(box[:, 1])]
                            top_points = sorted_box[:2][np.argsort(sorted_box[:2, 0])]
                            bottom_points = sorted_box[2:][np.argsort(sorted_box[2:, 0])[::-1]]
                            obb_points = np.vstack([top_points, bottom_points]).tolist()

                        # Normalize coordinates to [0, 1]
                        normalized_points = []
                        all_valid = True
                        for point in obb_points:
                            norm_x = point[0] / img_width
                            norm_y = point[1] / img_height
                            # Ensure values are within valid range [0, 1]
                            if not (0 <= norm_x <= 1 and 0 <= norm_y <= 1):
                                all_valid = False
                                break
                            normalized_points.extend([norm_x, norm_y])

                        # Create YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                        if all_valid and len(normalized_points) == 8:
                            coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_points])
                            yolo_line = f"{class_id} {coords_str}"
                            yolo_annotations.append(yolo_line)

                    # Skip if no valid annotations after filtering
                    if len(yolo_annotations) == 0:
                        skipped += 1
                        continue

                    # Save image
                    image_filename = f"{split_name}_{ds_name}_{idx:06d}.png"
                    image_path = os.path.join(yolo_dataset_dir, split_name, 'images', image_filename)
                    image.save(image_path)

                    # Save YOLO annotations
                    label_filename = f"{split_name}_{ds_name}_{idx:06d}.txt"
                    label_path = os.path.join(yolo_dataset_dir, split_name, 'labels', label_filename)
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))

                    successful += 1

                except Exception as e:
                    print(f"Error processing example {idx} ({ds_name}): {e}")
                    failed += 1

        print(f"\n{split_name.upper()} split processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")
    
    return yolo_dataset_dir

def verify_hf_dataset_splits():
    """
    Verify the splits from Hugging Face dataset
    """
    print("Verifying Hugging Face dataset splits...")
    dataset = load_dataset("rinabuoy/layout-data-khmer-1290-line-only")
    
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total examples: {len(split_data)}")
        print(f"  Features: {split_data.features}")
        
        # Sample first example
        if len(split_data) > 0:
            example = split_data[0]
            print(f"  Sample image size: {example['image'].size}")
            
            # Parse the annotation JSON string
            annotation_data = json.loads(example['annotation'])
            num_shapes = len(annotation_data.get('shapes', []))
            print(f"  Sample annotations: {num_shapes} objects")
            if num_shapes > 0:
                labels = [shape['label'] for shape in annotation_data['shapes']]
                print(f"  Labels in sample: {labels}")

#verify_hf_dataset_splits()


def create_data_yaml_for_combining():
    """
    Create data.yaml file for YOLO training with the COMBINING dataset
    """
    yolo_dataset_dir = os.path.join('/app/workspace', 'YOLO_DATASET_LINE')
    
    # Get label mapping
    label_mapping = get_label_mapping()
    # Create class names list in order of class IDs
    class_names = [''] * len(label_mapping)
    for label, class_id in label_mapping.items():
        class_names[class_id] = label
    
    yaml_content = f"""# COMBINING Dataset Configuration for YOLOv12
path: {yolo_dataset_dir}
train: train/images
val: validation/images
test: test/images

# Number of classes
nc: {len(label_mapping)}

# Class names
names: {class_names}
"""
    
    yaml_path = os.path.join('/app/workspace', 'data_combining_line.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at: {yaml_path}")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Classes: {class_names}")
    
    return yaml_path



#print("Loading dataset from Hugging Face to inspect structure...")
#dataset = load_dataset("rinabuoy/khmer-layout-data-standardized-with-line-annotations")
#
#print("\nDataset Information:")
#print("="*60)
#print(f"Available splits: {list(dataset.keys())}")
#
#for split_name in dataset.keys():
#    print(f"\n{split_name.upper()} Split:")
#    print(f"  Number of examples: {len(dataset[split_name])}")
#    print(f"  Features: {dataset[split_name].features}")
#    
#    # Show a sample
#    if len(dataset[split_name]) > 0:
#        sample = dataset[split_name][0]
#        print(f"\nSample from {split_name}:")
#        print(f"  Image size: {sample['image'].size}")
#        
#        # Parse annotation JSON
#        annotation_data = json.loads(sample['line_annotations'])
#        shapes = annotation_data.get('shapes', [])
#        print(f"  Number of objects: {len(shapes)}")
#        if len(shapes) > 0:
#            labels = [shape['label'] for shape in shapes]
#            print(f"  Labels in sample: {labels}")
#            print(f"  Sample points format: {shapes[0]['points'][:2] if len(shapes[0]['points']) > 0 else 'N/A'}")
#
#print("Starting COMBINING dataset preparation for YOLOv12 from Hugging Face...")
#print("="*60)

# Step 1: Convert Hugging Face dataset to YOLO format
yolo_dataset_dir = prepare_combining_dataset_from_hf()

print("\n" + "="*60)
# Step 2: Create data.yaml
#yaml_path = '/app/workspace/data_combining_line.yaml'#create_data_yaml_for_combining()
yaml_path = create_data_yaml_for_combining()
print("\n" + "="*60)
print("Dataset preparation complete!")
print(f"YOLO dataset location: {yolo_dataset_dir}")
print(f"Configuration file: {yaml_path}")


# Verify the dataset structure
def verify_dataset():
    """
    Verify that the dataset was created correctly
    """
    yolo_dataset_dir = os.path.join('/app/workspace', 'YOLO_DATASET_LINE')
    
    print("Dataset Structure Verification:")
    print("="*60)
    
    for split in ['train', 'validation', 'test']:
        images_path = os.path.join(yolo_dataset_dir, split, 'images')
        labels_path = os.path.join(yolo_dataset_dir, split, 'labels')
        
        if os.path.exists(images_path):
            num_images = len([f for f in os.listdir(images_path) if f.endswith('.png')])
            num_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {num_images}")
            print(f"  Labels: {num_labels}")
            
            # Sample a label file to show format
            if num_labels > 0:
                sample_label = os.listdir(labels_path)[0]
                print(f"  Sample label file: {sample_label}")
                with open(os.path.join(labels_path, sample_label), 'r') as f:
                    lines = f.readlines()[:3]  # Show first 3 annotations
                    print(f"  Sample annotations (first 3):")
                    for line in lines:
                        print(f"    {line.strip()}")

verify_dataset()


# Visualize sample annotations
def visualize_sample_annotation(split='train', num_samples=3):
    """
    Visualize a few sample images with their YOLO annotations
    """
    yolo_dataset_dir = os.path.join('/app/workspace', 'YOLO_DATASET_LINE')
    images_path = os.path.join(yolo_dataset_dir, split, 'images')
    labels_path = os.path.join(yolo_dataset_dir, split, 'labels')
    
    image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    label_mapping = get_label_mapping()
    # Reverse mapping: class_id -> label_name
    id_to_label = {v: k for k, v in label_mapping.items()}
    
    # Define colors for each class (BGR format for OpenCV)
    colors = [
        (255, 0, 0),    # text - red
        (0, 255, 0),    # image - green
        (0, 0, 255),    # section-header - blue
        (255, 255, 0),  # page-header - cyan
        (255, 0, 255),  # page-footer - magenta
        (0, 255, 255),  # title - yellow
        (128, 0, 128),  # list - purple
        (0, 128, 128),  # table - teal
        (128, 128, 0),  # figure - olive
    ]
    
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    
    for idx, img_file in enumerate(samples):
        # Read image
        img_path = os.path.join(images_path, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Read annotations
        label_file = img_file.replace('.png', '.txt')
        label_path = os.path.join(labels_path, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Convert from YOLO to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw rectangle
                        color = colors[class_id % len(colors)]
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label_name = id_to_label.get(class_id, f'class_{class_id}')
                        cv2.putText(img_rgb, label_name, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{img_file[:20]}...")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize samples
print("Visualizing sample annotations from training set:")
visualize_sample_annotation(split='train', num_samples=3)

yolov12_model_path = "/app/weights/la/yolo26x-obb.pt"
project_name = "khm_yolo26x_line"


#yolov12_model_path = "/app/weights/la/yolo12s.pt"
#project_name = "line_detection_yolo12s_obb_syn_id"

def train_yolov12(epochs=100, batch_size=8, project='/app/workspace/runs', name='yolo26x_line_khm'):
    # yolo26x-obb-khm.yaml: P2-level features + C2PSA at P3 + reg_max=4 for Khmer line detection
    model = YOLO("yolo26x-obb-khm.yaml").load(yolov12_model_path)
    model.train(data='/app/workspace/data_combining_line.yaml', epochs=epochs, batch=batch_size, imgsz=640, device='cuda', workers=4, save=True, save_period=20,
                        project=project, name=name,
                        flipud=0.0, fliplr=0.0,  # Khmer text must not be mirrored
                        mosaic=0.0,               # mosaic cuts text lines mid-way — harmful for line detection
                        degrees=3.0,              # scanned Khmer docs often have slight tilt (1-5°)
                        rect=True)                # portrait document pages — avoid padding waste
    model.val()
    print(model)
    print("YOLOv12 training completed on HRSC2016-MS.")


def infer_yolov12(image_path, model_weights="/app/workspace/runs/yolo26x_line_khm/weights/best.pt"):
    model = YOLO(model_weights)
    results = model(image_path)
    for result in results:
        img = cv2.imread(image_path)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
train_yolov12(name=project_name)

# Add this after the train_yolov12() call
from huggingface_hub import HfApi
import glob

def push_best_checkpoint_to_hf():
    """
    Push the best trained checkpoint to Hugging Face Hub
    """
    hf_api = HfApi()
    repo_id = "rinabuoy/khmer-ocr-checkpoints"
    
    # Find the latest training run directory
    runs_dir = '/app/workspace/runs'
    run_dirs = glob.glob(f"{runs_dir}/{project_name}*")
    
    if not run_dirs:
        print("No training runs found!")
        return
    
    # Get the most recent run
    latest_run = sorted(run_dirs)[-1]
    best_checkpoint = f"{latest_run}/weights/best.pt"
    
    if not os.path.exists(best_checkpoint):
        print(f"Best checkpoint not found at {best_checkpoint}")
        return
    
    print(f"Found best checkpoint: {best_checkpoint}")
    print(f"Pushing to Hugging Face Hub: {repo_id}")
    
    # Upload the best checkpoint
    hf_api.upload_file(
        path_or_fileobj=best_checkpoint,
        path_in_repo=f"{project_name}_best.pt",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"✓ Successfully pushed best checkpoint to {repo_id}")
    
    # Also upload the last checkpoint for backup
    last_checkpoint = f"{latest_run}/weights/last.pt"
    if os.path.exists(last_checkpoint):
        print(f"Pushing last checkpoint as backup...")
        hf_api.upload_file(
            path_or_fileobj=last_checkpoint,
            path_in_repo=f"{project_name}_last.pt",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✓ Successfully pushed last checkpoint to {repo_id}")

# Push the best checkpoint after training
push_best_checkpoint_to_hf()
#test_image = "/home/opencvuniv/yolov12_and_darknet/HRSC-YOLO/train/images/100001058.png"
#infer_yolov12(test_image)

#from ultralytics import YOLO
#
#model = YOLO("path_to_the_saved_model_weights_(can_be_found_at_runs/detect/trainx/weights/best.pt")
#
#metrics = model.val(data=data_yaml)
#print(metrics.box.map)
#print(metrics.box.map50)
#print(metrics.box.map75)
#
#from ultralytics import YOLO
#
#model = YOLO("path_to_the_saved_model_weights_(can_be_found_at_runs/detect/trainx/weights/best.pt")
#
#metrics = model.val(data=data_yaml)
#print(metrics.box.map)
#print(metrics.box.map50)
#print(metrics.box.map75)


