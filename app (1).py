import numpy as np
import cv2
import streamlit as st

# --- STREAMLIT UI INIT ---
st.set_page_config(page_title="Object Detection App", layout="centered")
st.title("🎯 Real-Time Object Detection App")
st.write("Upload an image below to detect and label objects using MobileNetV3.")

# --- INITIALIZATION & MODEL LOADING ---
thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress

classNames = []
with open('objects.txt','r') as f:
    classNames = f.read().splitlines()

# Distinct color mapping for drawing bounding boxes
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen-inference-graph.pb"
configPath = "ssd-mobilenet-v3-large-coco.pbtxt"

# Setup the OpenCV deep neural network model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# --- FILE UPLOADER COMPONENT ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded stream to a compatible OpenCV image matrix
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # CRITICAL SECURITY GUARD: Verify the image decoded successfully before running detection
    if img is not None:
        # Run inference
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        
        # Format the arrays into readable Python lists
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        # Apply Non-Maximum Suppression to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
        
        # --- DRAWING BOUNDING BOXES ---
        # Iterate over filtered detection indices and draw them directly onto the image
        if len(indices) > 0:
            for i in indices.flatten():
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                
                # Fetch class specific color identification and name metadata
                class_id = int(classIds[i])
                label = str(classNames[class_id - 1]).upper() if class_id <= len(classNames) else "UNKNOWN"
                score = confs[i]
                color = Colors[class_id % len(Colors)]
                
                # Draw visual bounding box overlay elements
                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
                cv2.putText(img, f"{label} {round(score*100, 1)}%", (x + 10, y + 30),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        
        # Convert color space array configuration to layout format optimized for Streamlit viewports (BGR -> RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Render the final composite image on the user pipeline output dashboard
        st.image(img_rgb, caption="Processed Image with Detections", use_container_width=True)
    else:
        st.error("Failed to process the uploaded image. Please try another file.")
else:
    st.info("Awaiting image upload. Please select a valid photo above to begin inference processing.")
