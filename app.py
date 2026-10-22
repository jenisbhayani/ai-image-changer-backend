import sys
from pathlib import Path

# Ensure root directory containing 'app' package is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import uvicorn
from app.main import app as fastapi_app

# Interactive dashboard displayed when opening the Space URL in a browser
with gr.Blocks(title="AI Image Changer Cloud API") as demo:
    gr.Markdown("# 🚀 AI Image Changer Cloud API")
    gr.Markdown(
        "Welcome! This Space runs the FastAPI backend for the **AI Image Changer** mobile app.\n\n"
        "### 📱 Endpoints for Mobile App:\n"
        "- **Health Check**: [`/api/v1/health`](/api/v1/health)\n"
        "- **Background Removal**: `POST /api/v1/remove-background`\n"
        "- **Interactive API Docs**: [`/docs`](/docs)\n"
    )

# Mount Gradio onto the existing FastAPI app
# FastAPI routes take priority, while the Gradio UI is served on root /
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    # Pass the app instance directly (NOT the string "app:app" to avoid collision with app/ directory)
    uvicorn.run(app, host="0.0.0.0", port=7860)
