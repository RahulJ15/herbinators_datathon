
import tempfile
from PIL import Image
import torch 
from yolov5.detect import run, get_image_path, parse_opt


import streamlit as st 




# Load YOLOv5 weights and model
weights = 'yolov5s.pt'  # You may need to adjust the path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = run(weights=weights, device=device, exist_ok=True)

# Streamlit app
st.title("Fruit Detection with YOLOv5")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        image_path = temp_file.name

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv5 detection
    opt = parse_opt()
    opt.input = image_path
    detections = run(**vars(opt))

    # Display detection results
    st.subheader("Detection Results:")
    
    # Get the number of detections
    num_detections = sum([len(det) for det in detections])
    st.write(f"Number of Detections: {num_detections}")

    # Display individual detections
    for i, det in enumerate(detections):
        st.write(f"Detection {i + 1}:")
        for bbox in det:
            st.write(f"Class: {model.names[int(bbox[5])]}, Confidence: {bbox[4]:.2f}")

    # Cleanup: Delete the temporary image file
    temp_file.close()
    st.success("Detection completed and temporary files cleaned up.")
