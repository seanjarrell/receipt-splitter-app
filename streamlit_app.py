import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

# --- Page Config ---
st.set_page_config(page_title="FinTech Receipt Splitter", page_icon="🧾", layout="centered")

# --- CSS for Mobile Optimization ---
st.markdown("""
<style>
    /* Make buttons bigger for touch targets */
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
    }
    /* Hide the default Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 1. Load Models (Cached) ---
@st.cache_resource
def load_models():
    """Loads text detection (YOLO) and extraction (Donut) models once."""
    print("Loading models...")
    # YOLO for Receipt Detection
    yolo_model = YOLO("yolov8n.pt")
    
    # Donut for Information Extraction
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Models loaded on {device}")
    return yolo_model, processor, model, device

try:
    with st.spinner("Initializing AI Brain... (This takes a moment)"):
        yolo_model, donut_processor, donut_model, device = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 2. Helper Functions ---

def extract_receipt_data(image_pil):
    """Runs the Donut model on a single receipt image to get JSON data."""
    # Prepare image for model
    pixel_values = donut_processor(image_pil, return_tensors="pt").pixel_values
    
    # Prepare prompt
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    
    # Generate output
    outputs = donut_model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=donut_model.config.decoder.max_position_embeddings,
    )
    
    # Decode output
    sequence = donut_processor.batch_decode(outputs)[0]
    sequence = sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
    clean_seq = re.sub(r"<.*?>", "", sequence, count=1).strip()
    
    return donut_processor.token2json(clean_seq)

def get_best_value(data, keys):
    """Safely retrieves values from nested JSON."""
    for key in keys:
        if key in data:
            return data[key]
        for k, v in data.items():
            if isinstance(v, dict) and key in v:
                return v[key]
    return None

def process_image(uploaded_file):
    """Main pipeline: Detect -> Crop -> Extract -> Display."""
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) # For PIL
    
    # 1. Detect Receipts
    results = yolo_model.predict(source=original_img, conf=0.15, classes=[73]) # 73 usually matches paper/documents in COCO
    
    detected_boxes = results[0].boxes
    
    if len(detected_boxes) == 0:
        st.warning("⚠️ No receipts detected. Try moving closer or improving lighting.")
        return

    st.success(f"✅ Found {len(detected_boxes)} receipts!")
    
    # 2. Loop through each detected receipt
    for i, box in enumerate(detected_boxes):
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Crop
        crop = original_rgb[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop)
        
        # 3. Extract Info (Donut)
        with st.status(f"Reading Receipt {i+1}...", expanded=False) as status:
            data = extract_receipt_data(crop_pil)
            status.update(label=f"Receipt {i+1} Processed", state="complete")
        
        # 4. Determine Filename
        store_name = get_best_value(data, ["nm", "store_nm", "brand", "company"]) or "Unknown"
        date = get_best_value(data, ["dt", "date", "issued_date"]) or "NoDate"
        
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(store_name))
        clean_date = re.sub(r'[^0-9-]', '', str(date)).replace("/", "-")
        
        filename = f"{clean_date}_{clean_name}_{i+1}.jpg"
        
        # 5. Display Result Card
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(crop_pil, use_column_width=True)
            
        with col2:
            st.markdown(f"**🏪 {store_name}**")
            st.markdown(f"📅 {date}")
            
            # Prepare download
            buf = io.BytesIO()
            crop_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"💾 Save Image",
                data=byte_im,
                file_name=filename,
                mime="image/jpeg",
                key=f"btn_{i}"
            )
            
            with st.expander("See Raw Data"):
                st.json(data)

# --- 3. UI Layout ---

st.title("📑 Receipt Splitter")
st.write("Take a photo of multiple receipts to split and name them automatically.")

# Mobile Camera Input
img_file_buffer = st.camera_input("Take a Picture")

if img_file_buffer is not None:
    process_image(img_file_buffer)

# Fallback for desktop testing (File Upload)
with st.expander("Or Upload Image (For Desktop Testing)"):
    uploaded_file = st.file_uploader("Upload Receipts", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
         process_image(uploaded_file)

st.write("---")
st.caption("FIN4901 FinTech Project | Streamlit Cloud Edition")

