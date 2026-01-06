import numpy as np
from scipy.spatial import KDTree
import random
# NOTE: Tkinter is required for the file dialog, and is usually built-in with Python.
import tkinter as tk
from tkinter import filedialog

# NOTE: OpenCV is required for video processing and particle detection
# You need to install it: pip install opencv-python
try:
    import cv2
except ImportError:
    print("Warning: OpenCV (cv2) not found. Using mock data generation.")
    cv2 = None


# --- CONFIGURATION ---

# Path to your video file or camera index (will be set by file dialog)
VIDEO_SOURCE = None 

# Region for the main column mask (Normalized coordinates 0 to 1)
# Use this to focus on the main column area and ignore pipes/background.
# E.g., a central region: (x_start, y_start, x_end, y_end)
COLUMN_MASK_NORMALIZED = (0.2, 0.05, 0.8, 0.95)

# Assume frame rate of your high-speed camera (in frames per second)
# CRITICAL: This MUST match your video's actual FPS for correct speed calculation.
FRAME_RATE = 100 
DELTA_T = 1.0 / FRAME_RATE # Time difference between frames (s)

# Column dimensions (initial guess, will be updated by video resolution)
COLUMN_HEIGHT = 800
COLUMN_WIDTH = 400

# Define Regions of Interest (ROI) for Top and Bottom (Normalized 0 to 1)
# These define the sections where speed measurements are averaged.
ROI_TOP_NORMALIZED = (0.0, 0.10) # From 0% to 10% of column height
ROI_BOTTOM_NORMALIZED = (0.90, 1.0) # From 90% to 100% of column height

# Maximum distance a particle is expected to travel between frames (pixels)
MAX_TRACKING_DISTANCE = 30 
# Particle detection threshold (adjust based on lighting/contrast)
BINARY_THRESHOLD = 120 


# --- HELPER FUNCTION FOR FILE SELECTION ---

def get_video_source():
    """
    Opens a file dialog to allow the user to select a video file.
    
    Returns: The path to the selected video file (string), or None if cancelled.
    """
    # Initialize Tkinter root window but immediately withdraw it to hide the main window
    root = tk.Tk()
    root.withdraw()
    
    print("Opening file selection dialog...")
    
    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select Bupsty Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    
    # Destroy the root window after use
    root.destroy()
    
    if file_path:
        print(f"Selected video file: {file_path}")
        return file_path
    else:
        print("File selection cancelled by user.")
        return None


# --- CORE PTV FUNCTIONS ---

def process_frame_for_particles(frame, width, height):
    """
    Detects particle centroids in a single video frame using image processing.
    
    Returns: A NumPy array of particle centroids (N, 2) in (x, y) coordinates.
    """
    if frame is None or cv2 is None:
        # Fallback to mock data if OpenCV is not available or frame is None
        num_particles = 100
        x = np.random.randint(0, width, num_particles)
        y = np.random.randint(0, height, num_particles)
        return np.column_stack((x, y))

    # 1. Apply Column Mask
    # Create a mask image (white for ROI, black otherwise)
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    
    x_start = int(COLUMN_MASK_NORMALIZED[0] * width)
    y_start = int(COLUMN_MASK_NORMALIZED[1] * height)
    x_end = int(COLUMN_MASK_NORMALIZED[2] * width)
    y_end = int(COLUMN_MASK_NORMALIZED[3] * height)

    cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # 2. Image Pre-processing
    gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Particle Segmentation (using simple thresholding, adjust BINARY_THRESHOLD)
    _, thresh = cv2.threshold(blurred, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    # Find centroids of contours (particles)
    for c in contours:
        # Filter small contours (noise). Adjust area minimum as needed.
        if cv2.contourArea(c) > 5: 
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
    
    return np.array(centroids)


def track_particles(prev_positions, current_positions, max_dist):
    """
    Matches particles between two consecutive frames using a nearest-neighbor 
    search limited by MAX_TRACKING_DISTANCE.
    
    Returns: A list of (prev_index, current_index) tuples for matched particles.
    """
    if prev_positions.size == 0 or current_positions.size == 0:
        return []

    # Use a KD-Tree for efficient nearest neighbor search in the current frame
    tree = KDTree(current_positions)
    
    matches = []
    
    # Iterate through all particles in the previous frame
    for i, p_prev in enumerate(prev_positions):
        # Find the closest particle in the current frame
        distance, j = tree.query(p_prev, k=1, distance_upper_bound=max_dist)
        
        # If a match is found within the max distance
        if j != tree.n:
            matches.append((i, j))
            
    return matches

def calculate_velocity_data(matched_pairs, prev_pos, curr_pos, delta_t):
    """
    Calculates the velocity vector for each matched particle.
    
    Returns: A list of dictionaries, where each dict contains position and velocity.
    """
    velocity_data = []
    
    for i_prev, i_curr in matched_pairs:
        p_prev = prev_pos[i_prev]
        p_curr = curr_pos[i_curr]
        
        # Displacement vector (dx, dy)
        displacement = p_curr - p_prev
        
        # Velocity vector (vx, vy)
        velocity = displacement / delta_t
        
        # Speed (scalar magnitude)
        speed = np.linalg.norm(velocity)
        
        velocity_data.append({
            'x': p_curr[0],
            'y': p_curr[1],
            'vx': velocity[0],
            'vy': velocity[1],
            'speed': speed
        })
        
    return velocity_data


def analyze_column_regions(all_velocity_data, height):
    """
    Separates calculated velocities into the top and bottom ROIs and 
    calculates the average speed for each.
    """
    top_speeds = []
    bottom_speeds = []
    
    # Calculate pixel boundaries based on normalized settings
    roi_top_end_pixel = int(height * ROI_TOP_NORMALIZED[1])
    roi_bottom_start_pixel = int(height * ROI_BOTTOM_NORMALIZED[0])
    
    for data in all_velocity_data:
        y = data['y']
        speed = data['speed']
        
        # Check if particle is in the TOP ROI (y < top_end)
        if y < roi_top_end_pixel:
            top_speeds.append(speed)
        
        # Check if particle is in the BOTTOM ROI (y > bottom_start)
        elif y > roi_bottom_start_pixel:
            bottom_speeds.append(speed)
            
    # Calculate averages
    avg_speed_top = np.mean(top_speeds) if top_speeds else 0
    avg_speed_bottom = np.mean(bottom_speeds) if bottom_speeds else 0
    
    print(f"\n--- ANALYSIS RESULTS ---")
    print(f"Total velocity measurements: {len(all_velocity_data)}")
    print(f"Measurements in TOP ROI (0 - {ROI_TOP_NORMALIZED[1]*100}% of height): {len(top_speeds)}")
    print(f"Measurements in BOTTOM ROI ({ROI_BOTTOM_NORMALIZED[0]*100}% - 100% of height): {len(bottom_speeds)}")
    
    print(f"\nAverage Particle Speed (TOP):    {avg_speed_top:,.2f} pixels/s")
    print(f"Average Particle Speed (BOTTOM): {avg_speed_bottom:,.2f} pixels/s")

    # Determine the faster region
    if avg_speed_top > avg_speed_bottom * 1.05: # 5% tolerance
        print("CONCLUSION: Particles are SIGNIFICANTLY faster at the TOP.")
    elif avg_speed_bottom > avg_speed_top * 1.05:
        print("CONCLUSION: Particles are SIGNIFICANTLY faster at the BOTTOM.")
    elif avg_speed_top > 0 or avg_speed_bottom > 0:
        print("CONCLUSION: Speeds are relatively similar between the top and bottom regions.")
    else:
        print("CONCLUSION: No particle movement detected or insufficient data.")


# --- MAIN ANALYSIS LOOP ---

def run_ptv_analysis():
    """Reads video, processes frames, and performs PTV analysis."""
    global VIDEO_SOURCE

    if cv2 is None:
        print("FATAL ERROR: OpenCV is not installed. Cannot process video.")
        print("Running mock data simulation instead for demonstration...")
        # Fallback to the original mock simulation if cv2 is not available
        return run_mock_simulation(num_frames=20)
    
    # Step 1: Get video source from user input
    # If VIDEO_SOURCE is None, prompt the user with the file dialog
    if VIDEO_SOURCE is None:
        video_path = get_video_source()
        if video_path is None:
            print("Video analysis aborted.")
            return
        VIDEO_SOURCE = video_path

    # Step 2: Open Video Capture
    print(f"Attempting to open video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{VIDEO_SOURCE}'.")
        print("Please check the path or ensure your camera is connected (if using index 0).")
        print("Running mock data simulation instead for demonstration...")
        return run_mock_simulation(num_frames=20)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    global COLUMN_WIDTH, COLUMN_HEIGHT
    COLUMN_WIDTH, COLUMN_HEIGHT = width, height
    
    print(f"Video resolution detected: {width}x{height} pixels.")
    
    prev_positions = np.array([])
    all_velocity_measurements = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If reading was unsuccessful (end of video or error), break
        if not ret:
            print("Finished reading video frames.")
            break
        
        # 1. Detect particles in the current frame
        current_positions = process_frame_for_particles(frame, width, height)
        
        if prev_positions.size > 0:
            # 2. Track particles from previous to current frame
            matched_pairs = track_particles(prev_positions, current_positions, MAX_TRACKING_DISTANCE)
            
            # 3. Calculate velocities
            velocities = calculate_velocity_data(matched_pairs, prev_positions, current_positions, DELTA_T)
            all_velocity_measurements.extend(velocities)
            
            # Optional: Display the tracking results (uncomment for visual debugging)
            # frame_display = frame.copy()
            # for data in velocities:
            #     cv2.circle(frame_display, (int(data['x']), int(data['y'])), 3, (0, 255, 0), -1)
            #     cv2.arrowedLine(frame_display, 
            #                     (int(data['x']) - int(data['vx']*DELTA_T), int(data['y']) - int(data['vy']*DELTA_T)),
            #                     (int(data['x']), int(data['y'])),
            #                     (255, 0, 0), 1)
            # cv2.imshow('PTV Tracking', frame_display)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Prepare for next frame
        prev_positions = current_positions
        frame_count += 1
        
        # For live stream compatibility or periodic reporting, analyze every N frames
        if frame_count % 50 == 0 and len(all_velocity_measurements) > 0:
            print(f"\n--- Progress Report (Frame {frame_count}) ---")
            analyze_column_regions(all_velocity_measurements, height)
            
            # Clear data collection if this is a continuous stream analysis
            # all_velocity_measurements = [] 
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Final analysis after processing all frames
    if len(all_velocity_measurements) > 0:
        print("\n--- FINAL ANALYSIS ---")
        analyze_column_regions(all_velocity_measurements, COLUMN_HEIGHT)
    else:
        print("\nNo particle movement detected or insufficient data for final analysis.")


# --- MOCK SIMULATION LOOP (Kept for safe fallback/testing without OpenCV) ---
def run_mock_simulation(num_frames):
    """Simulates the PTV process over a sequence of frames."""
    global COLUMN_HEIGHT, COLUMN_WIDTH
    print("Starting PTV simulation (mock data)...")
    
    # Define pixel boundaries for ROIs based on current mock height
    roi_top_end_pixel = int(COLUMN_HEIGHT * ROI_TOP_NORMALIZED[1])
    
    mock_positions = []
    pos_frame0 = process_frame_for_particles(None, COLUMN_WIDTH, COLUMN_HEIGHT) 
    mock_positions.append(pos_frame0)
    
    for i in range(1, num_frames):
        prev_pos = mock_positions[-1]
        new_pos = []
        
        for x, y in prev_pos:
            # Random movement base
            dx = random.uniform(-5, 5) 
            dy = random.uniform(2, 12) # Overall downward bias (positive Y is down)
            
            # Add a bias to the top region (Top ROI is 0 to roi_top_end_pixel)
            if y < roi_top_end_pixel:
                dx += random.uniform(-10, 10) # More erratic side movement at top
                dy += 15 # Significantly faster downward movement at top
                
            new_pos.append([x + dx, y + dy])
        
        new_pos_np = np.array(new_pos)
        new_pos_np[:, 0] = np.clip(new_pos_np[:, 0], 0, COLUMN_WIDTH - 1)
        new_pos_np[:, 1] = np.clip(new_pos_np[:, 1], 0, COLUMN_HEIGHT - 1)

        mock_positions.append(new_pos_np)

    all_velocity_measurements = []
    
    for i in range(len(mock_positions) - 1):
        prev_pos = mock_positions[i]
        curr_pos = mock_positions[i+1]
        
        matches = track_particles(prev_pos, curr_pos, MAX_TRACKING_DISTANCE)
        velocities = calculate_velocity_data(matches, prev_pos, curr_pos, DELTA_T)
        all_velocity_measurements.extend(velocities)
        
    analyze_column_regions(all_velocity_measurements, COLUMN_HEIGHT)


if __name__ == "__main__":
    run_ptv_analysis()