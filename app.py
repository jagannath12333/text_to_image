# app.py
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from transformers import CLIPImageProcessor
from peft import PeftModel
from PIL import Image
import os

# --- Configuration ---
PRETRAINED_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS_DIR = "MarketingModel_LoRA"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Model Loading ---
# Load the model once when the script starts
try:
    safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
        torch_dtype=torch.float16,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        use_safetensors=True
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_DIR)
    pipe = pipe.to(DEVICE)
    
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()
    
    print("Model loaded successfully.")
    model_loaded = True
except Exception as e:
    print(f"Error loading the model: {e}")
    model_loaded = False

# --- Image Generation Function for Gradio ---
def generate_image(prompt, num_inference_steps, guidance_scale, seed):
    """
    Generates an image from a text prompt using the globally loaded pipeline.
    """
    if not model_loaded:
        raise gr.Error("Model could not be loaded. Please check the logs.")
        
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        return image
    except Exception as e:
        raise gr.Error(f"An error occurred during image generation: {e}")

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ AI-Powered Marketing Image Generator")
    gr.Markdown("This tool uses a fine-tuned Stable Diffusion model to create professional product photography from your text descriptions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_text = gr.Textbox(
                label="Describe Your Product",
                value="professional product photography of a coffee brand, studio lighting, high quality, 4k, on a clean background",
                lines=4
            )
            inference_steps = gr.Slider(
                label="Inference Steps",
                minimum=10,
                maximum=100,
                step=1,
                value=50
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=20.0,
                step=0.1,
                value=7.5
            )
            seed_value = gr.Number(
                label="Seed",
                value=42,
                precision=0
            )
            generate_button = gr.Button("✨ Generate Image", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_text, inference_steps, guidance_scale, seed_value],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()
