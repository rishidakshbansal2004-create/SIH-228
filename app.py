"""
TrustCV Verification & Security API - Hugging Face Spaces Entrypoint
Runs on ZeroGPU (NVIDIA A10G 24GB + 16GB RAM) with full FastAPI route compatibility.
"""

try:
    import spaces
    print("SUCCESS: ZeroGPU spaces module loaded successfully!", flush=True)

    @spaces.GPU
    def gpu_accelerator_status():
        return "🛡️ TrustCV Hardware Accelerator Active (ZeroGPU NVIDIA A10G)"
except Exception as e:
    print(f"NOTICE: ZeroGPU spaces not active ({e})", flush=True)

    def gpu_accelerator_status():
        return "🛡️ TrustCV Engine Active (Standard CPU 16 GB RAM)"

import os
import sys
from pathlib import Path
import gradio as gr

# Ensure local directories resolve correctly
repo_root = Path(__file__).resolve().parent
app_dir = repo_root / "trusted_cv_model_integrity_final"
mirad_dir = repo_root / "MIRAD"

sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(mirad_dir))

from api_server import app as fastapi_app

# Gradio wrapper for the root page (satisfies ZeroGPU supervisor & provides live status)
with gr.Blocks(title="TrustCV Verification & Security API") as demo:
    gr.Markdown("# 🛡️ TrustCV Verification & Security API")
    gr.Markdown("FastAPI Backend is running live with high-memory execution for Phase 4 B3D verification.")
    
    with gr.Row():
        status_box = gr.Textbox(label="Hardware Accelerator Status", value="Ready")
        check_btn = gr.Button("Initialize Hardware Accelerator", variant="primary")
        check_btn.click(fn=gpu_accelerator_status, inputs=[], outputs=status_box)

    gr.Markdown("### Direct Endpoints:")
    gr.Markdown("- **Health Check**: [`/api/health`](/api/health)")
    gr.Markdown("- **Phase 4 B3D Analysis**: `/api/verify/model/phase4`")
    gr.Markdown("- **Swagger Documentation**: [`/docs`](/docs)")
    gr.Markdown("- **OpenAPI Specification**: [`/openapi.json`](/openapi.json)")

# Hook all FastAPI endpoints directly into Gradio's internal FastAPI server
orig_create_app = gr.routes.App.create_app

def custom_create_app(*args, **kwargs):
    server_app = orig_create_app(*args, **kwargs)
    for route in fastapi_app.routes:
        server_app.routes.append(route)
    return server_app

gr.routes.App.create_app = custom_create_app

# Enable Gradio queue for ZeroGPU integration
demo.queue()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    print(f"Launching Gradio with TrustCV endpoints on 0.0.0.0:{port} (ssr_mode=False)...", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
    )
