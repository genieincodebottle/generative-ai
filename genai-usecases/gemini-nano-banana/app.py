import streamlit as st
from google import genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key  
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Set page config
st.set_page_config(
    page_title="Gemini-2.5 Flash Image Generator", 
    page_icon="🎨", 
    layout="wide"
)

# App title and description
st.title("🎨 Gemini-2.5 Flash Image Generator (nano banana)")
st.markdown("Generate and process images")

# Sidebar for API key
with st.sidebar:
    st.header("🔐 Configuration")
    if api_key:
        st.success("Google API Key is set")
        st.text_input(
            "API Key",
            value=api_key,
            type="password",
            help="Get your API key from Google AI Studio",
            disabled=True
        )
    else:
        api_key = st.text_input("Enter your Google API Key", type="password", 
                            help="Get your API key from Google AI Studio")
        
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("API Key configured!")
        else:
            st.warning("Please enter your Google AI API Key to continue")

# Main interface
if api_key:
    try:
        # Initialize the client
        client = genai.Client()
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📝 Input")
            
            # Text prompt input
            prompt = st.text_area(
                "Enter your image generation prompt:",
                value="A majestic dog dressed in a sleek tuxedo, seated at an opulent dining table, savoring gourmet cuisine while attentive waiters serve under the warm glow of a grand golden-lit hall adorned with luxurious décor.",
                height=100,
                help="Describe what you want to generate or analyze"
            )
            
            # File uploader for reference image
            uploaded_file = st.file_uploader(
                "Upload a reference image (optional):",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                help="Upload an image to use as reference for generation or analysis"
            )
            
            # Display uploaded image
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Reference Image", width='stretch')
            
            # Model selection
            model_name = st.selectbox(
                "Select Gemini Model:",
                options=[
                    "gemini-2.5-flash-image-preview"
                ],
                help="Choose the Gemini model for processing"
            )
            
            # Generate button
            generate_button = st.button("Generate/Process", type="primary", width='stretch')
        
        with col2:
            st.header("📤 Output")
            
            if generate_button and prompt:
                with st.spinner("Processing with Gemini-2.5 Flash Image..."):
                    try:
                        # Prepare contents for the API call
                        contents = [prompt]
                        if uploaded_file is not None:
                            contents.append(image)
                        
                        # Make API call
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                        )
                        
                        # Process response
                        text_outputs = []
                        generated_images = []
                        
                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                text_outputs.append(part.text)
                            elif part.inline_data is not None:
                                generated_image = Image.open(BytesIO(part.inline_data.data))
                                generated_images.append(generated_image)
                        
                        # Display text outputs
                        if text_outputs:
                            st.subheader("📄 Text Response")
                            for i, text in enumerate(text_outputs):
                                st.write(text)
                        
                        # Display generated images
                        if generated_images:
                            st.subheader("🖼️ Generated Images")
                            for i, img in enumerate(generated_images):
                                st.image(img, caption=f"Generated Image {i+1}", width='stretch')
                                
                                # Download button for each image
                                buf = BytesIO()
                                img.save(buf, format="PNG")
                                st.download_button(
                                    label=f"📥 Download Image {i+1}",
                                    data=buf.getvalue(),
                                    file_name=f"generated_image_{i+1}.png",
                                    mime="image/png"
                                )
                        
                        if not text_outputs and not generated_images:
                            st.info("No output generated. Try a different prompt or model.")
                            
                    except Exception as e:
                        st.error(f"Error processing request: {str(e)}")
                        st.info("Make sure your API key is valid and you have access to the selected model.")
            
            elif generate_button and not prompt:
                st.warning("Please enter a prompt to generate content.")

    except Exception as e:
        st.error(f"Error initializing Gemini client: {str(e)}")
        st.info("Please check your API key and internet connection.")

else:
    st.info("👈 Please enter your Google AI API Key in the sidebar to get started.")
    
    with st.expander("ℹ️ How to get your API Key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://aistudio.google.com/)
        2. Sign in with your Google account
        3. Click on "Get API Key" 
        4. Create a new API key or use an existing one
        5. Copy and paste it in the sidebar
        """)