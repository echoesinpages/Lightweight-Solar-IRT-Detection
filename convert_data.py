import os
import glob
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# 1. Define relative paths for your local Windows machine
input_dir = 'PVEL-AD_raw'
output_dir = 'dataset_yolo'

# Using parentheses instead of brackets to prevent UI formatting bugs
classes = (
    "crack", "star_crack", "finger_interruption", "black_core",
    "thick_line", "scratch", "fragment", "corner",
    "printing_error", "horizontal_dislocation", "vertical_dislocation",
    "short_circuit"
)
class_to_id = {cls: i for i, cls in enumerate(classes)}

# 2. Create directories
for split in ('train', 'val'):
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 3. Completely Fixed Mathematical Conversion (Zero Brackets Used)
def convert_box(size, box):
    # Unpack tuples directly to completely avoid formatting bugs
    image_w, image_h = size
    xmin, xmax, ymin, ymax = box
    
    # Calculate normalization scales
    dw = 1.0 / image_w
    dh = 1.0 / image_h
    
    # Calculate YOLO format requirements
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Apply normalization
    x = x_center * dw
    y = y_center * dh
    w = width * dw
    h = height * dh
    
    return (x, y, w, h)

print("Converting XML files to YOLO format...")

# Specifically target the Annotations folder
xml_files = glob.glob(os.path.join(input_dir, 'Annotations', '*.xml'))

images_and_labels = list()

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract the filename and point directly to the JPEGImages folder
    base_name = os.path.basename(xml_file)
    img_name = base_name.replace('.xml', '.jpg')
    img_path = os.path.join(input_dir, 'JPEGImages', img_name)
    
    if not os.path.exists(img_path):
        img_path = os.path.join(input_dir, 'JPEGImages', base_name.replace('.xml', '.png'))
        if not os.path.exists(img_path):
            continue
            
    size_node = root.find('size')
    if size_node is None:
        continue
        
    img_width = float(size_node.find('width').text)
    img_height = float(size_node.find('height').text)
    
    yolo_labels = list()
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name not in class_to_id:
            continue
        cls_id = class_to_id.get(cls_name)
        xmlbox = obj.find('bndbox')
        
        # Extract bounding box boundaries
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
             
        # Execute the fixed conversion
        bb = convert_box((img_width, img_height), b)
        yolo_labels.append(f"{cls_id} {' '.join(map(str, bb))}")
        
    if len(yolo_labels) > 0:
        images_and_labels.append((img_path, yolo_labels))

if len(images_and_labels) == 0:
    print("ERROR: Still finding 0 images. Please check your folder names.")
    exit()

train_data, val_data = train_test_split(images_and_labels, test_size=0.15, random_state=42)

def process_split(data, split_name):
    for img_path, labels in data:
        base_name = os.path.basename(img_path)
        txt_name = base_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        shutil.copy(img_path, os.path.join(output_dir, 'images', split_name, base_name))
        
        with open(os.path.join(output_dir, 'labels', split_name, txt_name), 'w') as f:
            f.write('\n'.join(labels))

process_split(train_data, 'train')
process_split(val_data, 'val')

# Generate the YAML file using an absolute path required by Windows
abs_output_dir = os.path.abspath(output_dir).replace('\\', '/')
yaml_content = f"""path: {abs_output_dir}
train: images/train
val: images/val

names:
  0: crack
  1: star_crack
  2: finger_interruption
  3: black_core
  4: thick_line
  5: scratch
  6: fragment
  7: corner
  8: printing_error
  9: horizontal_dislocation
  10: vertical_dislocation
  11: short_circuit
"""

with open(os.path.join(output_dir, 'pvel_ad.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"Dataset conversion complete! Successfully copied and converted {len(images_and_labels)} images.")