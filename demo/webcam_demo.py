import os
import cv2
import numpy as np
import torch
import time

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor

# Global variables for mouse interaction
points = []
labels = []
drawing_bbox = False
bbox_start = None
bbox_end = None
current_obj_id = 1
mode = "point"  # "point" or "bbox"

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points, labels, drawing_bbox, bbox_start, bbox_end, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "point":
            # Add positive point
            points.append([x, y])
            labels.append(1)
            print(f"Added positive point at ({x}, {y})")
        elif mode == "bbox" and not drawing_bbox:
            # Start drawing bbox
            drawing_bbox = True
            bbox_start = [x, y]
            print(f"Started bbox at ({x}, {y})")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == "point":
            # Add negative point
            points.append([x, y])
            labels.append(0)
            print(f"Added negative point at ({x}, {y})")
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == "bbox" and drawing_bbox:
            # Update bbox end point while dragging
            bbox_end = [x, y]
    
    elif event == cv2.EVENT_LBUTTONUP:
        if mode == "bbox" and drawing_bbox:
            # Finish drawing bbox
            bbox_end = [x, y]
            drawing_bbox = False
            print(f"Finished bbox from ({bbox_start[0]}, {bbox_start[1]}) to ({bbox_end[0]}, {bbox_end[1]})")

def main():
    global points, labels, drawing_bbox, bbox_start, bbox_end, current_obj_id, mode
    
    # Load model
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_t_512.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, vos_optimized=True)
    
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create window and set mouse callback
    window_name = "SAM2 Webcam Demo"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    if_init = False
    tracking_i = 0
    paused = True
    out_obj_ids = []
    out_mask_logits = []
    
    print("\n=== SAM2 Webcam Demo ===")
    print("Controls:")
    print("  p: Toggle between point and bbox mode")
    print("  c: Clear all points/bbox")
    print("  space: Pause/resume tracking")
    print("  n: New object ID")
    print("  a: Apply current points/bbox")
    print("  q: Quit")
    print("Mouse:")
    print("  Left click: Add positive point or start/end bbox")
    print("  Right click: Add negative point")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
        
        # Flip the frame horizontally for a more natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        width, height = rgb_frame.shape[:2][::-1]
        
        # Draw current points and bbox on display frame
        display_frame = frame.copy()
        
        # Draw points
        for i, (point, label) in enumerate(zip(points, labels)):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, Red for negative
            cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, color, -1)
        
        # Draw bbox
        if mode == "bbox" and drawing_bbox and bbox_start and bbox_end:
            cv2.rectangle(display_frame, 
                         (int(bbox_start[0]), int(bbox_start[1])), 
                         (int(bbox_end[0]), int(bbox_end[1])), 
                         (255, 0, 0), 2)
        
        # Show current mode and object ID
        cv2.putText(display_frame, f"Mode: {mode.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Object ID: {current_obj_id}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Paused" if paused else "Tracking", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if paused else (0, 255, 0), 2)
        
        if not if_init and not paused:
            # Initialize with first frame
            predictor.load_first_frame(rgb_frame)
            if_init = True
            print("Initialized tracking with first frame")
        
        elif if_init and not paused:
            start_time = time.time()
            
            # Track objects in current frame
            out_obj_ids, out_mask_logits = predictor.track(rgb_frame)
            tracking_i += 1
            
            # Visualize masks
            all_mask = np.zeros((height, width, 3), dtype=np.uint8)
            all_mask[..., 1] = 255  # Set saturation to max
            
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                
                hue = (i + 3) / (len(out_obj_ids) + 3) * 255
                all_mask[out_mask[..., 0] == 255, 0] = hue
                all_mask[out_mask[..., 0] == 255, 2] = 255
            
            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
            display_frame = cv2.addWeighted(display_frame, 1, all_mask, 0.5, 0)
            
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow(window_name, display_frame)
        
        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
            
        elif key == ord('p'):
            # Toggle between point and bbox mode
            mode = "bbox" if mode == "point" else "point"
            print(f"Switched to {mode} mode")
            
        elif key == ord('c'):
            # Clear all points/bbox
            points = []
            labels = []
            bbox_start = None
            bbox_end = None
            drawing_bbox = False
            if_init = False
            paused = True
            predictor.reset_state()
            
            print("Cleared all points/bbox")
            
        elif key == ord(' '):
            # Pause/resume tracking
            paused = not paused
            print("Tracking paused" if paused else "Tracking resumed")
            
        elif key == ord('n'):
            # New object ID
            current_obj_id += 1
            print(f"New object ID: {current_obj_id}")
            
        elif key == ord('a'):
            # Apply current points/bbox
            if if_init:
                predictor.add_conditioning_frame(rgb_frame)
                
                if mode == "point" and points:
                    # Apply points
                    point_array = np.array(points, dtype=np.float32)
                    label_array = np.array(labels, dtype=np.int32)
                    
                    predictor.add_new_prompt_during_track(
                        point=point_array,
                        labels=label_array,
                        obj_id=current_obj_id,
                        if_new_target=False,
                        clear_old_points=True,
                    )
                    print(f"Applied {len(points)} points for object ID {current_obj_id}")
                    
                elif mode == "bbox" and bbox_start and bbox_end:
                    # Apply bbox
                    bbox = np.array([bbox_start, bbox_end], dtype=np.float32)
                    
                    predictor.add_new_prompt_during_track(
                        bbox=bbox,
                        obj_id=current_obj_id,
                        if_new_target=False,
                        clear_old_points=True,
                    )
                    print(f"Applied bbox for object ID {current_obj_id}")
                
                # Clear after applying
                points = []
                labels = []
                bbox_start = None
                bbox_end = None
                
            else:
                # First initialization with points or bbox
                predictor.load_first_frame(rgb_frame)
                if_init = True
                
                ann_frame_idx = 0
                
                if mode == "point" and points:
                    # Apply points
                    point_array = np.array(points, dtype=np.float32)
                    label_array = np.array(labels, dtype=np.int32)
                    
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, 
                        obj_id=current_obj_id, 
                        points=point_array, 
                        labels=label_array
                    )
                    print(f"Initialized with {len(points)} points for object ID {current_obj_id}")
                    
                elif mode == "bbox" and bbox_start and bbox_end:
                    # Apply bbox
                    bbox = np.array([bbox_start, bbox_end], dtype=np.float32)
                    
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, 
                        obj_id=current_obj_id, 
                        bbox=bbox
                    )
                    print(f"Initialized with bbox for object ID {current_obj_id}")
                
                # Clear after applying
                points = []
                labels = []
                bbox_start = None
                bbox_end = None
                
                # Start tracking automatically after initialization
                paused = False
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()