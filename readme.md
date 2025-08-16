# 🚀 Marketing Image Generator — README (Complete, production-ready steps)

Thanks — you uploaded the LoRA files. I’ve expanded the README to include **every missing step** you asked for: where to put files, how to push to GitHub, how to upload to Hugging Face Spaces (including Git LFS for `.safetensors`), how to rebuild/restart, how to verify the model+LoRA loaded, and a troubleshooting checklist with exact fixes for the errors you saw.

Copy this `README.md` into the root of your project repository.

---

# 🎨 Marketing Image Generator (Stable Diffusion + LoRA)

Generate brand-aligned marketing and advertising images using a fine-tuned Stable Diffusion model (LoRA). This repository includes inference code (Gradio UI), instructions to train or reuse LoRA, and deployment steps to GitHub / Hugging Face Spaces.

---

## 📁 Repo layout (what you should have uploaded)

```
marketing-image-generator/
├─ app.py                       # Inference app (Gradio)
├─ requirements.txt             # Exact pinned dependencies (see below)
├─ MarketingModel_LoRA/         # LoRA adapter files (must be here)
│  ├─ adapter_model.safetensors
│  └─ adapter_config.json
├─ README.md                    # This file
├─ .gitattributes               # (for Git LFS)
└─ train.py (optional)          # Training script if you want to fine-tune
```

> **Important:** `adapter_model.safetensors` is large — track it with **Git LFS** (instructions below).

---

## ✅ Recommended `requirements.txt` (paste exactly)

Use this in your Space or when running locally (matches the app code and is known to be stable):

```txt
--find-links https://download.pytorch.org/whl/cu118/torch_stable.html

torch==2.1.2+cu118
torchvision==0.16.2+cu118

diffusers==0.27.2
transformers==4.37.2
accelerate==0.26.0
huggingface-hub==0.25.2

peft==0.6.2
safetensors==0.4.5
gradio==4.44.0
Pillow==10.4.0
numpy<2.0
```

---

## 🔧 What I added to `app.py`

* A small shim to avoid the `cached_download` import crash.
* Automatic CPU vs GPU detection.
* CPU-optimized defaults (384px on CPU, safety-checker disabled for speed).
* LoRA support — your `MarketingModel_LoRA/` will be detected and loaded automatically (if `peft` available).
* Clear startup logging to verify versions.

---

## 🗂️ How to upload & commit files (Git + Git LFS)

If you use GitHub + Hugging Face Spaces, follow these steps locally:

1. Initialize repo (if not done):

```bash
git init
git checkout -b main
```

2. Install Git LFS and track safetensors:

```bash
git lfs install
git lfs track "*.safetensors"
echo "*.safetensors filter=lfs diff=lfs merge=lfs -text" > .gitattributes
git add .gitattributes
```

3. Add files and push to GitHub:

```bash
git add .
git commit -m "Initial commit — app + LoRA"
# Add GitHub remote (replace URL)
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

> If you prefer to upload files directly in Hugging Face Spaces web UI, upload `adapter_model.safetensors` and `adapter_config.json` into `MarketingModel_LoRA/`. But if you push via git, use Git LFS for the `.safetensors`.

---

## ☁️ Deploy to Hugging Face Spaces (step-by-step)

1. Create a **new Space** → choose **Gradio** as SDK.
2. Connect repository (either push to GitHub and link, or push directly to the Space repo).
3. Ensure `requirements.txt` is present in root.
4. Ensure `MarketingModel_LoRA/adapter_model.safetensors` and `adapter_config.json` are present in the repo (or uploaded via the Files tab).
5. **Restart/Rebuild** the Space:

   * In Space UI: *Settings* → *Restart with rebuild* (recommended when requirements changed).
6. Monitor build logs (Files → View logs). Wait for build to complete.

---

## 🔍 Verifying the deployment (what to check after rebuild)

1. Open the Space logs (first lines) — you should see the startup banner:

   ```
   torch: <version> | diffusers: <version> | transformers: <version> | accelerate: <version> | huggingface_hub: <version> | device: cpu/cuda, dtype: <dtype>
   ```
2. Look for:

   * `Model loaded successfully.` or `LoRA loaded successfully!`.
   * If LoRA was found but `peft` missing: `Found LoRA files but peft is not installed; continuing without LoRA.` — install `peft` in requirements to fix.
3. Test the UI by entering a prompt and clicking **Generate**.
4. If you want to confirm LoRA effects: switch off LoRA (temporarily rename the folder) to compare output with/without LoRA.

---

## ✅ Example test prompt (marketing-focused)

Use this to test that your model produces marketing visuals:

**Prompt**

```
professional advertising photo of a confident young person holding a premium coffee cup, cinematic studio lighting, high-end magazine style, product-focused composition, realistic skin texture, 4k
```

**Negative prompt**

```
low quality, blurry, bad anatomy, extra fingers, cartoon, watermark, text
```

---

## ⚠️ Troubleshooting — common errors & fixes

### 1) `ImportError: cannot import name 'cached_download'`

* Cause: `diffusers`/`huggingface_hub` mismatch.
* Fix: Use the recommended `requirements.txt`. The provided `app.py` includes a shim that prevents crash, but correct pinned packages are best.

### 2) `ModuleNotFoundError: No module named 'torch.distributed.device_mesh'`

* Cause: `accelerate` needs newer torch/distributed API.
* Fix: Use the pinned `accelerate==0.26.0` + `torch==2.1.2+cu118` (or upgrade torch and use matching accelerate).

### 3) `ModuleNotFoundError: Could not import CLIPTextModel` / `AutoModel`

* Cause: `transformers` incompatible version.
* Fix: Use `transformers==4.37.2` (pinned above).

### 4) LoRA not applied

* Check that `MarketingModel_LoRA/` contains `adapter_model.safetensors` and `adapter_config.json`.
* Confirm `peft` is installed (in `requirements.txt`).
* Check logs for `Loaded PEFT LoRA from: ...`

### 5) Very slow generation (long `processing | XXs`)

* If `device: cpu` in logs → you are on CPU Basic (slow).

  * Lower `IMG_SIZE` to 320–384.
  * Lower `num_inference_steps` to 10–20.
  * Optionally switch Space hardware to **ZeroGPU** or **T4 small** (if you have GPU quota).
* If `device: cuda` and still slow, try `enable_xformers_memory_efficient_attention()` (in app it is used where available).

---

## 🧪 Local testing (run locally before deploying)

1. Create and activate a venv:

```bash
python -m venv .venv
source .venv/bin/activate       # mac/linux
.venv\Scripts\activate          # windows
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open [http://localhost:7860/](http://localhost:7860/) and test.

---

## 🔐 Hugging Face Hub (optional)

If you want to upload your LoRA adapter to Hugging Face Model Hub:

1. `huggingface-cli login` (enter token).
2. `git lfs install` in model repo.
3. Push LoRA model or use `huggingface_hub` functions to upload.

When loading private models in a Space, use environment variable `HUGGING_FACE_HUB_TOKEN` (set in Space settings) and call `.from_pretrained(model_id, use_auth_token=True)` if needed.

---

## ✅ Checklist before marking “Done”

* [ ] `app.py`, `requirements.txt`, `MarketingModel_LoRA/` present in repo
* [ ] `.gitattributes` configured and `.safetensors` tracked with Git LFS
* [ ] Commit and push to GitHub (or directly upload to Space)
* [ ] Space: *Restart with rebuild*
* [ ] Verify startup banner and `Model loaded successfully.` log
* [ ] Run a marketing prompt and inspect outputs
* [ ] If you want GPU: request or switch to ZeroGPU/T4 in Space Settings

---

## 📄 Licensing & Credits

* License this repo as **MIT** (or choose your preferred license).
* Credit the base model (`runwayml/stable-diffusion-v1-5`) and any datasets you used.

---

