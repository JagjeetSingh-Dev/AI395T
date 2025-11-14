import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # Load the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Get detections for the specific view
    if view_index >= len(info["detections"]):
        return []
    
    frame_detections = info["detections"][view_index]
    kart_objects = []
    
    # Calculate scaling factors from original to target image size
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Extract kart objects (class_id == 1)
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Only process karts (class_id == 1)
        if class_id != 1:
            continue
        
        # Scale coordinates to target image size
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Filter out karts that are out of sight (outside image boundaries)
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Calculate center of the bounding box
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Get kart name from the karts list
        kart_name = None
        for kart in info["karts"]:
            if kart["id"] == track_id:
                kart_name = kart["kart"]
                break
        
        # If kart name not found, skip this detection
        if kart_name is None:
            continue
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y)
        })
    
    # Identify the center kart (closest to image center)
    if kart_objects:
        image_center_x = img_width / 2
        image_center_y = img_height / 2
        
        # Calculate distance from each kart to image center
        min_distance = float('inf')
        center_kart_idx = 0
        
        for i, kart in enumerate(kart_objects):
            cx, cy = kart["center"]
            distance = ((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                center_kart_idx = i
        
        # Mark the center kart
        for i, kart in enumerate(kart_objects):
            kart["is_center_kart"] = (i == center_kart_idx)
    
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # Load the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Extract and return the track name
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []
    
    # Extract kart objects and track info
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # If no karts detected, return empty list
    if not karts:
        return qa_pairs
    
    # Find the ego car (center kart)
    ego_kart = None
    for kart in karts:
        if kart["is_center_kart"]:
            ego_kart = kart
            break
    
    # If no ego car found, return empty list
    if ego_kart is None:
        return qa_pairs
    
    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_kart["kart_name"]
    })
    
    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })
    
    # 3. Track information question
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })
    
    # Get ego car position
    ego_x, ego_y = ego_kart["center"]
    
    # Initialize counters for relative positions
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0
    
    # 4. Relative position questions for each non-ego kart
    for kart in karts:
        if kart["is_center_kart"]:
            continue  # Skip ego car
        
        kart_x, kart_y = kart["center"]
        kart_name = kart["kart_name"]
        
        # Determine left/right position
        if kart_x < ego_x:
            left_right = "left"
            left_count += 1
        else:
            left_right = "right"
            right_count += 1
        
        # Determine front/behind position (lower y = front in image coordinates)
        if kart_y < ego_y:
            front_behind = "front"
            front_count += 1
        else:
            front_behind = "behind"
            behind_count += 1
        
        # Add left/right question
        qa_pairs.append({
            "question": f"Is {kart_name} to the left or right of the ego car?",
            "answer": left_right
        })
        
        # Add front/behind question
        qa_pairs.append({
            "question": f"Is {kart_name} in front of or behind the ego car?",
            "answer": front_behind
        })
    
    # 5. Counting questions
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count)
    })
    
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind_count)
    })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs})


if __name__ == "__main__":
    main()
