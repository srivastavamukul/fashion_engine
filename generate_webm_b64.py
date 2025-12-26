
import cv2
import numpy as np
import base64
import os

def generate():
    # Create a simple animation
    height, width = 64, 64
    filename = "temp_valid.webm"
    
    # VP80 is usually supported in cv2 out of the box for .webm
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
    
    if not out.isOpened():
        print("ERROR: VP80 codec not valid. Trying VP90...")
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (width, height))
        
    if not out.isOpened():
        print("ERROR: VP90 codec not valid. Trying MJPG (AVI) as fallback...")
        # MJPG in AVI is widely supported but not in modern <video> tags consistently
        # But let's try to fail if we can't make webm
        return

    for i in range(10):
        # Color cycle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [(i * 25) % 255, 100, 200] 
        out.write(frame)
        
    out.release()
    
    # Read binary and encode
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"Generated webm size: {size}")
        
        with open(filename, "rb") as f:
            data = f.read()
        
        b64 = base64.b64encode(data).decode('utf-8')
        with open("valid_webm.txt", "w", encoding="utf-8") as f:
            f.write(b64)
        print("SUCCESS: Wrote valid_webm.txt")
        
        # Cleanup
        # os.remove(filename)
    else:
        print("FAIL: No file generated")

if __name__ == "__main__":
    generate()
