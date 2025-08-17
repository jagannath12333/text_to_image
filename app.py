import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Title
st.title("🎨 Marketing Image Generator")
st.write("Generate marketing visuals using your custom LoRA model.")

# Cache pipeline so it's not reloaded every button click
@st.cache_resource
def load_pipeline():
    base_model = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # Load your LoRA weights
    pipe.load_lora_weights("marketing_model")
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

# Prompt input
prompt = st.text_area("Enter your marketing prompt:", "High-quality coffee advertisement poster, minimal design")

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        try:
            pipe = load_pipeline()
            image: Image.Image = pipe(prompt).images[0]

            # Show result
            st.image(image, caption="Generated Marketing Image", use_container_width=True)

            # Save image to buffer for download
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="marketing_image.png",
                mime="image/png",
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
