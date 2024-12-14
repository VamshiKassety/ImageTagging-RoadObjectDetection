import os
import json

def generate_label_files(json_filename):
    # Map categories to ClassID
    category_to_classid = {
        "traffic sign": 1,
        "traffic light": 2,
        "car": 3,
        "rider": 4,
        "motorcycle": 5,
        "pedestrian": 6,
        "bus": 7,
        "truck": 8,
        "bicycle": 9,
        "other vehicle": 10,
        "train": 11,
        "trailer": 12,
        "other person": 13
    }

    # Load the JSON file
    with open(json_filename, 'r') as file:
        data = json.load(file)

    # Ensure the labels directory exists
    labels_dir = "../labels"
    os.makedirs(labels_dir, exist_ok=True)

    # Process each image's data
    for image_data in data:
        image_name = image_data['name']
        label_file_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

        # Skip file creation if it already exists
        if os.path.exists(label_file_path):
            print(f"Skipping {label_file_path}, already exists.")
            continue

        # Generate the label content
        label_lines = []
        for label in image_data.get('labels', []):
            category = label['category']
            box = label['box2d']

            if category in category_to_classid:
                class_id = category_to_classid[category]
                x_center = (box['x1'] + box['x2']) / 2
                y_center = (box['y1'] + box['y2']) / 2
                width = box['x2'] - box['x1']
                height = box['y2'] - box['y1']

                # Format line
                line = f"{class_id} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}"
                label_lines.append(line)

        # Write the content to the label file
        with open(label_file_path, 'w') as label_file:
            label_file.write("\n".join(label_lines))

        print(f"Created {label_file_path}")

# Example usage:
# generate_label_files('path_to_json_file.json')