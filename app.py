import os
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image

# Optional: LoRA (PEFT)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ------------------- Config -------------------
BASE_MODEL = os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
LORA_DIR   = os.getenv("LORA_DIR", "./MarketingModel_LoRA")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
IMG_SIZE = 384 if DEVICE == "cpu" else 512  # smaller on CPU for speed

# ------------------- Load pipeline ------------
@st.cache_resource  # cache model so it doesn’t reload each time
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        safety_checker=None,
        feature_extractor=None,
        use_safetensors=True,
    )

    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            st.warning(f"Note: xFormers not enabled ({e})")
    else:
        pipe.enable_attention_slicing()

    pipe = pipe.to(DEVICE)

    # ---- Load LoRA if present ----
    if os.path.isdir(LORA_DIR):
        has_adapter = os.path.exists(os.path.join(LORA_DIR, "adapter_model.safetensors"))
        if has_adapter and _HAS_PEFT:
            try:
                pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
                st.info(f"✅ Loaded LoRA from: {LORA_DIR}")
            except Exception as e:
                st.warning(f"LoRA load failed (using base model): {e}")
        elif has_adapter and not _HAS_PEFT:
            st.warning("Found LoRA files but `peft` is not installed.")
        else:
            st.info("No LoRA files found, using base model.")
    else:
        st.info("LoRA dir not found, using base model.")

    return pipe

pipe = load_model()

# ------------------- Streamlit UI ----------------
st.title("🖼️ Face/Photo Generator (SD v1.5 + optional LoRA)")

prompt = st.text_area(
    "Prompt",
    "portrait photo of a person, soft lighting, 50mm, shallow depth of field, highly detailed, 4k",
    height=100,
)

negative_prompt = st.text_area(
    "Negative Prompt (optional)",
    "low quality, worst quality, bad anatomy, extra fingers, blurry",
    height=60,
)

steps = st.slider("Inference Steps", 10, 50, 20)
guidance = st.slider("Guidance Scale", 1.0, 12.0, 7.5, step=0.1)
seed = st.number_input("Seed (>=0 fixed, -1 random)", value=42, step=1)

if st.button("✨ Generate"):
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    with st.spinner("Generating image..."):
        try:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=IMG_SIZE,
                height=IMG_SIZE,
                generator=generator,
            )
            image: Image.Image = out.images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Generation error: {e}")
