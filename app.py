import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title
st.title("🎨 Marketing Image Generator")
st.write("Generate marketing visuals using your custom LoRA model.")

# Prompt input
prompt = st.text_area("Enter your marketing prompt:", "High-quality coffee advertisement poster, minimal design")

# Generate button
if st.button("Generate Image"):
    with st.spinner("Loading model and generating image..."):
        try:
            # Load base Stable Diffusion model
            base_model = "runwayml/stable-diffusion-v1-5"

            pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            # Load your LoRA adapter
            pipe.load_lora_weights("marketing_model")

            # Use GPU if available
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            # Generate image
            image: Image.Image = pipe(prompt).images[0]

            # Show result
            st.image(image, caption="Generated Marketing Image", use_container_width=True)

            # Allow download
            st.download_button(
                label="Download Image",
                data=image.tobytes(),
                file_name="marketing_image.png",
                mime="image/png",
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
