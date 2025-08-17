import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

st.set_page_config(page_title="Marketing AI Generator", layout="centered")
st.title("🎨 Marketing Image Generator (CPU Mode)")
st.write("Generate marketing visuals using your custom LoRA model (CPU-only).")

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe.load_lora_weights("marketing_model")
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

prompt = st.text_area("Enter your marketing prompt:", "Coffee ad poster, minimal design")
steps = st.slider("Inference Steps", 10, 25, 15)
guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image (CPU may take 1-3 minutes)..."):
            image: Image.Image = pipe(
                prompt, num_inference_steps=steps, guidance_scale=guidance
            ).images[0]

            st.image(image, caption="Generated Marketing Image", use_column_width=True)
