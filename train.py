from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import torch

# Base model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

pipe.unet = get_peft_model(pipe.unet, lora_config)

# Training loop (simplified)
for step, batch in enumerate(dataloader):
    loss = pipe.training_step(batch)
    loss.backward()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
    optimizer.step()
    optimizer.zero_grad()

pipe.save_pretrained("MarketingModel_LoRA")
