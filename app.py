import os
import warnings

# --- Pre-patch for cached_download removal (for older diffusers) ----
try:
    import huggingface_hub as hfh
    if not hasattr(hfh, "cached_download"):
        def cached_download(*args, **kwargs):
            return hfh.hf_hub_download(*args, **kwargs)
        hfh.cached_download = cached_download
except Exception:
    pass

import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

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
print(f"Starting… torch={torch.__version__}, device={DEVICE}, dtype={DTYPE}")

model_loaded = False
pipe = None

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        safety_checker=None,      # disable safety checker for speed
        feature_extractor=None,   # disable extra feature extractor
        use_safetensors=True,
    )

    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Note: xFormers not enabled ({e})")
    else:
        pipe.enable_attention_slicing()

    pipe = pipe.to(DEVICE)

    # ---- Load LoRA if present ----
    if os.path.isdir(LORA_DIR):
        has_adapter = os.path.exists(os.path.join(LORA_DIR, "adapter_model.safetensors"))
        if has_adapter and _HAS_PEFT:
            try:
                pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
                print(f"Loaded PEFT LoRA from: {LORA_DIR}")
            except Exception as e:
                print(f"LoRA load failed (continuing without LoRA): {e}")
        elif has_adapter and not _HAS_PEFT:
            print("Found LoRA files but peft is not installed; continuing without LoRA.")
        else:
            print("No LoRA files found, continuing with base model.")
    else:
        print("LoRA dir not found; continuing with base model.")

    model_loaded = True
    print("✅ Model loaded successfully.")

except Exception as e:
    print("❌ Error during model load:", repr(e))
    model_loaded = False


# ------------------- Inference fn -------------
def generate_image(prompt, negative_prompt, steps, guidance, seed):
    if not model_loaded:
        raise gr.Error("Model failed to load. Check logs for details.")

    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=IMG_SIZE,
            height=IMG_SIZE,
            generator=generator,
        )
        return out.images[0]
    except Exception as e:
        raise gr.Error(f"Generation error: {e}")


# ------------------- Gradio UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Face/Photo Generator (SD v1.5 + optional LoRA)")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="portrait photo of a person, soft lighting, 50mm, shallow depth of field, highly detailed, 4k",
                lines=4,
            )
            negative = gr.Textbox(
                label="Negative Prompt (optional)",
                value="low quality, worst quality, bad anatomy, extra fingers, blurry",
                lines=2,
            )
            steps = gr.Slider(10, 50, value=20, step=1, label="Inference Steps")
            guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.1, label="Guidance Scale")
            seed = gr.Number(value=42, precision=0, label="Seed (>=0 fixed, -1 random)")
            go = gr.Button("✨ Generate", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="Output", type="pil")

    def _wrap_seed(s):
        s = int(s)
        return None if s < 0 else s

    go.click(
        fn=lambda p, n, st, g, s: generate_image(p, n, st, g, _wrap_seed(s)),
        inputs=[prompt, negative, steps, guidance, seed],
        outputs=out_img,
    )

if __name__ == "__main__":
    demo.launch()
