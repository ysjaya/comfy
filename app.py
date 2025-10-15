import subprocess
import modal
from pathlib import Path

# === 1. DAFTAR MODEL FINAL (SESUAI WORKFLOW ANDA) ===
# Hanya model yang benar-benar ada di workflow Anda yang akan diunduh.
MODEL_REGISTRY = {
    "diffusion_models": {
        "wan2.2_ti2v_5B_fp16.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"},
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"},
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        "wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors"},
        "wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors"},
        "wan2.2_s2v_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"},
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"},
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"},
    },
    "vae": {
        "wan2.2_vae.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/vae/wan2.2_vae.safetensors"},
        "wan_2.1_vae.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/vae/wan_2.1_vae.safetensors"},
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {"repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
    },
    "loras": {
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"},
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"},
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"},
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"},
    },
    "audio_encoders": {
        "wav2vec2_large_english_fp16.safetensors": {"repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "filename": "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"}
    }
}

# === 2. FUNGSI UNTUK MENGUNDUH MODEL ===
def hf_download():
    from huggingface_hub import hf_hub_download
    base_model_path = Path("/root/comfy/ComfyUI/models")
    for model_type, models in MODEL_REGISTRY.items():
        target_dir = base_model_path / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Memproses tipe model: {model_type}")
        for filename, details in models.items():
            target_path = target_dir / filename
            if not target_path.exists():
                print(f"  - Mengunduh {filename}...")
                cached_model_path = hf_hub_download(repo_id=details["repo_id"], filename=details["filename"], cache_dir="/cache")
                print(f"  - Membuat link untuk {filename}...")
                subprocess.run(f"ln -s '{cached_model_path}' '{target_path}'", shell=True, check=True)
            else:
                print(f"  - Melewati {filename}, sudah ada.")
    print("\n[+] Semua model berhasil diunduh dan ditautkan!")

# === 3. DEFINISI MODAL IMAGE (TERMASUK DOWNLOAD WORKFLOW) ===
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget") # Tambahkan wget
    .pip_install("comfy-cli==1.4.1", force_build=True)
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.54",
        force_build=True,
    )
    .run_commands(
        "comfy node install ComfyUI-VideoHelperSuite",
        "comfy node install was-node-suite-comfyui",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        force_build=True,
    )
    # --- Perintah untuk mengunduh workflow Anda ---
    .run_commands(
        "mkdir -p /root/workflows && cd /root/workflows && "
        "wget https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_5B_ti2v.json && "
        "wget https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_t2v.json && "
        "wget https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_s2v.json && "
        "wget https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_i2v.json && "
        "wget https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_fun_camera.json",
        force_build=True,
    )
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# Jalankan fungsi download model saat membangun image
image = image.run_function(
    hf_download,
    volumes={"/cache": vol},
    timeout=4200
)

# === 4. APLIKASI MODAL UNTUK MENJALANKAN UI ===
app = modal.App(name="comfyui-wan-workflows", image=image)

@app.function(
    gpu="L40S",
    volumes={"/cache": vol},
    scaledown_window=300,
    allow_concurrent_inputs=10,
)
@modal.web_server(8000, startup_timeout=180)
def ui():
    # Salin workflow dari /root/workflows ke direktori ComfyUI agar bisa langsung diakses
    # Ini opsional, tapi memudahkan
    subprocess.Popen("cp /root/workflows/*.json /root/comfy/ComfyUI/", shell=True)
    # Jalankan server ComfyUI
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
