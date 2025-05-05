import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from langchain_groq import ChatGroq
import requests

# Set page config
st.set_page_config(
    page_title="Image Insight AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            color: #2c3e50;
            text-align: center;
            padding: 1rem 0;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .upload-box {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .result-box {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image to base64."""
    try:
        if isinstance(image_path, Image.Image):
            buffered = BytesIO()
            image_path.save(buffered, format='JPEG')
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = image.convert("RGB")
                buffered = BytesIO()
                image.save(buffered, format='JPEG')
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                raise ValueError(f"Failed to fetch image from URL: {image_path}")
        else:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def initialize_groq_client():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        api_key='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',
    )

# Sidebar
with st.sidebar:
    st.title("🖼️ Image Insight AI")
    st.markdown("""
        **How to use:**
        1. Upload an image or enter a URL
        2. Enter your question about the image
        3. Click 'Analyze Image'
    """)
    st.markdown("---")
    st.markdown("""
        Powered by:
        - [Groq](https://groq.com/)
        - [Llama 3](https://llama.meta.com/)
    """)

# Main content
st.title("Image Insight AI")
st.caption("Get AI-powered analysis of your images in seconds")

# Input options
tab1, tab2 = st.tabs(["Upload Image", "Image URL"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], 
                                   help="Upload an image for analysis")

with tab2:
    image_url = st.text_input("Or enter an image URL", 
                            placeholder="https://example.com/image.jpg",
                            help="Paste a URL to an image")

# Custom prompt
prompt = st.text_area("What would you like to know about the image?",
                     value="Describe the image in one sentence",
                     help="Ask any question about the uploaded image")

# Process button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)

# Display and processing
if analyze_btn:
    image_to_process = None
    image_source = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        image_source = "upload"
    elif image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image_to_process = Image.open(BytesIO(response.content))
                image_source = "url"
            else:
                st.error(f"Failed to fetch image from URL. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error loading image from URL: {str(e)}")
    else:
        st.warning("Please upload an image or enter an image URL")
    
    if image_to_process:
        with st.spinner("Analyzing image..."):
            try:
                # Display image
                st.subheader("Your Image")
                st.image(image_to_process, use_column_width=True)
                
                # Encode image
                b64_image = encode_image_to_base64(image_to_process)
                
                if b64_image:
                    # Initialize Groq client
                    lvlm = initialize_groq_client()
                    
                    # Prepare messages
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                            ]
                        }
                    ]
                    
                    # Get response
                    response = lvlm.invoke(messages)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    with st.expander("View detailed response", expanded=True):
                        st.markdown(f"**Question:** {prompt}")
                        st.markdown("---")
                        st.markdown(f"**Answer:** {response.content}")
                        st.markdown("---")
                        st.caption(f"Model used: meta-llama/llama-4-scout-17b-16e-instruct")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")