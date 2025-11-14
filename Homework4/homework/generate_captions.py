from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    from .generate_qa import extract_kart_objects, extract_track_info
    
    captions = []
    
    # Extract kart objects and track info
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # If no karts detected, return empty list
    if not karts:
        return captions
    
    # Find the ego car (center kart)
    ego_kart = None
    for kart in karts:
        if kart["is_center_kart"]:
            ego_kart = kart
            break
    
    # If no ego car found, return empty list
    if ego_kart is None:
        return captions
    
    # 1. Ego car caption
    captions.append(f"{ego_kart['kart_name']} is the ego car.")
    
    # 2. Counting caption
    captions.append(f"There are {len(karts)} karts in the scenario.")
    
    # 3. Track name caption
    captions.append(f"The track is {track_name}.")
    
    # Get ego car position
    ego_x, ego_y = ego_kart["center"]
    
    # 4. Relative position captions for each non-ego kart
    for kart in karts:
        if kart["is_center_kart"]:
            continue  # Skip ego car
        
        kart_x, kart_y = kart["center"]
        kart_name = kart["kart_name"]
        
        # Determine position relative to ego car
        # Horizontal position
        if kart_x < ego_x:
            horizontal = "left"
        else:
            horizontal = "right"
        
        # Vertical position (lower y = front in image coordinates)
        if kart_y < ego_y:
            vertical = "front"
        else:
            vertical = "behind"
        
        # Create position description
        position = f"to the {horizontal} and {vertical}"
        
        captions.append(f"{kart_name} is {position} of the ego car.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
