import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Marketing & Advertising AI 📣", layout="centered")
st.title("📣 AI Marketing Content Generator")
st.write("Generate unique, brand-specific visuals for your marketing campaigns using AI.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"  # base model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Load your LoRA adapters
    adapter_path = "MarketingModel_LoRA"  # folder inside your repo
    pipe.load_lora_weights(adapter_path)

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# ----------------------------
# User Input
# ----------------------------
prompt = st.text_area("Enter your campaign idea 🎯", 
                      placeholder="Example: A modern coffee shop ad with a smiling young woman holding a branded coffee cup")

steps = st.slider("Inference Steps", 10, 50, 25)
guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

if st.button("✨ Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating your custom marketing visual..."):
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
            st.image(image, caption="Generated Marketing Creative", use_column_width=True)
            st.success("✅ Done! You can right-click to save your ad image.")

