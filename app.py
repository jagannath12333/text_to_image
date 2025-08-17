import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# ----------------------------
# App Title
# ----------------------------
st.set_page_config(page_title="Marketing Image Generator", layout="centered")
st.title("🎨 AI Marketing Image Generator")
st.write("Generate marketing visuals using your custom LoRA model (CPU optimized).")

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,  # CPU friendly
    )

    # Load your LoRA adapter
    pipe.load_lora_weights("marketing_model")  # folder with adapter_model.safetensors & adapter_config.json

    # CPU mode
    pipe = pipe.to("cpu")

    # Enable memory efficient inference
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention(False)  # CPU, so no xformers

    return pipe

pipe = load_model()

# ----------------------------
# User Input
# ----------------------------
prompt = st.text_area(
    "Enter your marketing prompt:",
    "High-quality coffee advertisement poster, minimal design"
)

steps = st.slider("Inference Steps", 10, 30, 20)
guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

# ----------------------------
# Generate Image
# ----------------------------
if st.button("✨ Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image on CPU... (may take 1-2 minutes)"):
            with torch.inference_mode():
                image: Image.Image = pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance
                ).images[0]

            st.image(image, caption="Generated Marketing Image", use_column_width=True)

            # Allow download
            st.download_button(
                label="Download Image",
                data=image.tobytes(),
                file_name="marketing_image.png",
                mime="image/png"
            )
