# AI Image Changer - Cloud BiRefNet FastAPI Backend

This backend provides cloud-based high-resolution AI background removal using **BiRefNet** ([ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)), licensed under the **MIT License** for commercial use. It is designed for seamless, one-click deployment to **Hugging Face Spaces** (Docker or Python Space) or any cloud container runtime.

---

## Architecture & Endpoints

- `GET /` - Root health/discovery endpoint.
- `GET /api/v1/health` - Health check, model status, device detection (CUDA vs CPU).
- `GET /api/v1/model-info` - Model metadata, license, max dimensions, and input resolution.
- `POST /api/v1/remove-background` - Multipart upload (`image` field), EXIF orientation normalization, BiRefNet segmentation inference, and transparent RGBA PNG response.

---

## Local Development

### 1. Create and Activate Virtual Environment
```bash
cd backend
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
```
Visit `http://localhost:7860/docs` in your browser to test the interactive Swagger UI.

---

## Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK (Blank).
3. Set Space hardware:
   - **Free Tier (CPU Basic - 2 vCPU, 16 GB RAM)**: Uses `ZhengPeng7/BiRefNet_lite` by default. Takes ~4–10 seconds per image.
   - **GPU Tier (e.g. T4 small)**: Sets inference time to ~300ms per image.
4. Push the contents of the `backend/` folder to your Hugging Face Space repository:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
   git add .
   git commit -m "Deploy BiRefNet FastAPI backend"
   git push -u origin main --force
   ```
5. Once built, your Space URL will be:
   `https://<your-username>-<your-space-name>.hf.space`
6. Set this URL in your mobile app `.env`:
   `EXPO_PUBLIC_CLOUD_API_URL=https://<your-username>-<your-space-name>.hf.space`

---

## Free Hosting Limitations & Quota Handling
- Free Hugging Face Spaces sleep after 48 hours of inactivity. The first request will trigger a cold boot (~15–40 seconds). The mobile app gracefully displays "Connecting to cloud AI..." and handles timeouts.
- Free CPU instances have 16 GB RAM. Large images (>2500px) are automatically scaled proportionally to prevent out-of-memory errors.
