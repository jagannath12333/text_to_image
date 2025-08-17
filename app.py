import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("🎨 Custom Text-to-Image Generator")

# Load your fine-tuned model
@st.cache_resource
def load_model():
    model_path = "./"  # your fine-tuned model folder (contains model_index.json, etc.)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")  # Streamlit Cloud has no GPU
    return pipe

pipe = load_model()

# User input
prompt = st.text_area("Enter your prompt:", "A scenic view of mountains during sunset")

if st.button("Generate Image"):
    with st.spinner("Generating... (may take 1-3 minutes on CPU)"):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
