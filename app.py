"""
TrustCV Verification & Security API - Hugging Face Spaces Entrypoint
Mounts the TrustCV FastAPI backend onto Gradio for zero-billing 16 GB RAM hosting.
"""

try:
    import spaces
    print("SUCCESS: ZeroGPU spaces module loaded successfully!", flush=True)

    @spaces.GPU
    def gpu_accelerator_status():
        return "🛡️ TrustCV Accelerator Active (ZeroGPU A10G)"
except Exception as e:
    print(f"NOTICE: ZeroGPU spaces not active ({e})", flush=True)

    def gpu_accelerator_status():
        return "🛡️ TrustCV Engine Active (Standard CPU 16 GB RAM)"

import sys
from pathlib import Path
import uvicorn
import gradio as gr

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

# Enable Gradio queue for ZeroGPU integration
demo.queue()

# Mount Gradio at root with SSR disabled (prevents Node.js from binding port 7860)
app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    import os
    import subprocess
    print("=== STARTUP DIAGNOSTICS ===", flush=True)
    print(f"PORT: {os.environ.get('PORT')}", flush=True)
    print(f"GRADIO_SERVER_PORT: {os.environ.get('GRADIO_SERVER_PORT')}", flush=True)
    print(f"SPACES_ZERO_GPU: {os.environ.get('SPACES_ZERO_GPU')}", flush=True)
    try:
        print("=== RUNNING PROCESSES ===", flush=True)
        print(subprocess.check_output(["ps", "-ef"], text=True), flush=True)
    except Exception as e:
        print(f"ps error: {e}", flush=True)
    try:
        print("=== LISTENING PORTS ===", flush=True)
        print(subprocess.check_output("netstat -tlpn 2>/dev/null || ss -tlpn 2>/dev/null", shell=True, text=True), flush=True)
    except Exception as e:
        print(f"netstat error: {e}", flush=True)

    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    print(f"Binding uvicorn to 0.0.0.0:{port}...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
