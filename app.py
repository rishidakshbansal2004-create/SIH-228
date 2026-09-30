"""
TrustCV Verification & Security API - Hugging Face Spaces Entrypoint
Mounts the TrustCV FastAPI backend onto Gradio for zero-billing 16 GB RAM hosting.
"""

import sys
from pathlib import Path
import uvicorn
import gradio as gr

try:
    import spaces

    @spaces.GPU
    def gpu_accelerator_status():
        return "🛡️ TrustCV Accelerator Active (ZeroGPU A10G)"
except Exception:
    def gpu_accelerator_status():
        return "🛡️ TrustCV Engine Active (Standard CPU 16 GB RAM)"

# Ensure local directories resolve correctly
repo_root = Path(__file__).resolve().parent
app_dir = repo_root / "trusted_cv_model_integrity_final"
mirad_dir = repo_root / "MIRAD"

sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(mirad_dir))

from api_server import app as fastapi_app

# Gradio wrapper for the root page (keeps the Space active & gives a clean dashboard)
with gr.Blocks(title="TrustCV Verification & Security API") as demo:
    gr.Markdown("# 🛡️ TrustCV Verification & Security API")
    gr.Markdown("FastAPI Backend is running live with high-memory execution for Phase 4 B3D verification.")
    
    with gr.Row():
        status_box = gr.Textbox(label="Accelerator Status", value="Ready")
        check_btn = gr.Button("Initialize Hardware Accelerator", variant="primary")
        check_btn.click(fn=gpu_accelerator_status, inputs=[], outputs=status_box)

    gr.Markdown("### Direct Endpoints:")
    gr.Markdown("- **Health Check**: [`/api/health`](/api/health)")
    gr.Markdown("- **Phase 4 B3D Analysis**: `/api/verify/model/phase4`")
    gr.Markdown("- **Swagger Documentation**: [`/docs`](/docs)")
    gr.Markdown("- **OpenAPI Specification**: [`/openapi.json`](/openapi.json)")

# Mount Gradio at root while leaving all existing /api/* FastAPI routes intact
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
