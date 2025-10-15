"""
Production-Ready ComfyUI Wan 2.2 Video Generation API
- Pure JSON workflow system (no hardcoded workflows)
- Proper error handling and logging
- Health checks and monitoring
- Scalable architecture
- Easy workflow management
"""

import modal
import json
import base64
import time
import os
import subprocess
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import requests

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Workflow configuration
WORKFLOW_URLS = {
    "t2v_14B": "https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_t2v.json",
    "i2v_14B": "https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_i2v.json",
    "s2v_14B": "https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_s2v.json",
    "camera_14B": "https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_14B_fun_camera.json",
    "ti2v_5B": "https://raw.githubusercontent.com/ysjaya/comfy/main/video_wan2_2_5B_ti2v.json",
}

# Model registry
MODEL_REGISTRY = {
    "diffusion_models": {
        "wan2.2_ti2v_5B_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
        },
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_s2v_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
        },
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        },
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        },
    },
    "vae": {
        "wan2.2_vae.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/vae/wan2.2_vae.safetensors"
        },
        "wan_2.1_vae.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/vae/wan_2.1_vae.safetensors"
        },
    },
    "text_encoders": {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        },
    },
    "loras": {
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
        },
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
        },
        "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
        },
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
        },
    },
    "audio_encoders": {
        "wav2vec2_large_english_fp16.safetensors": {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"
        }
    }
}

# Constants
COMFYUI_PORT = 8000
COMFYUI_HOST = "127.0.0.1"
SERVER_STARTUP_TIMEOUT = 60
WORKFLOW_TIMEOUT = 300  # 5 minutes
POLL_INTERVAL = 2
MAX_POLL_ATTEMPTS = 150  # 5 minutes with 2s interval

# ============================================================================
# 2. WORKFLOW MAPPING SYSTEM
# ============================================================================

@dataclass
class WorkflowMapping:
    """Defines how to map request parameters to workflow nodes"""
    workflow_file: str
    node_mappings: Dict[str, Dict[str, str]]  # {param_name: {node_id: input_key}}
    required_files: List[str] = None  # List of file types needed (e.g., ['image', 'audio'])
    
    def __post_init__(self):
        if self.required_files is None:
            self.required_files = []

# Workflow mapping registry - EASY TO EDIT!
WORKFLOW_MAPPINGS = {
    "t2v_14B": WorkflowMapping(
        workflow_file="t2v_14B.json",
        node_mappings={
            "prompt": {"89": "text"},  # Positive prompt
            "negative_prompt": {"72": "text"},  # Negative prompt
            "seed": {"81": "seed", "78": "seed"}  # Both KSamplers
        },
        required_files=[]
    ),
    "i2v_14B": WorkflowMapping(
        workflow_file="i2v_14B.json",
        node_mappings={
            "image": {"97": "image"},  # LoadImage node
            "prompt": {"93": "text"},  # Positive prompt
            "seed": {"86": "seed"}  # KSampler
        },
        required_files=["image"]
    ),
    "s2v_14B": WorkflowMapping(
        workflow_file="s2v_14B.json",
        node_mappings={
            "image": {"52": "image"},  # LoadImage node
            "audio": {"58": "audio"},  # LoadAudio node
            "prompt": {"6": "text"},  # Positive prompt
            "seed": {"3": "seed"}  # KSampler
        },
        required_files=["image", "audio"]
    ),
    "camera_14B": WorkflowMapping(
        workflow_file="camera_14B.json",
        node_mappings={
            "image": {"79": "image"},  # LoadImage node
            "prompt": {"81": "text"},  # Positive prompt
            "camera_motion": {"87": "camera_motion"},  # WanCameraEmbedding
            "seed": {"71": "seed"}  # KSamplerAdvanced
        },
        required_files=["image"]
    ),
    "ti2v_5B": WorkflowMapping(
        workflow_file="ti2v_5B.json",
        node_mappings={
            "image": {"56": "image"},  # LoadImage node
            "prompt": {"6": "text"},  # Positive prompt
            "seed": {"3": "seed"}  # KSampler
        },
        required_files=["image"]
    ),
}

# ============================================================================
# 3. ENUMS & STATUS
# ============================================================================

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CameraMotion(str, Enum):
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    PAN_LEFT = "Pan Left"
    PAN_RIGHT = "Pan Right"
    TILT_UP = "Tilt Up"
    TILT_DOWN = "Tilt Down"
    STATIC = "Static"

# ============================================================================
# 4. PYDANTIC MODELS
# ============================================================================

class BaseWorkflowRequest(BaseModel):
    """Base request model with common parameters"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for generation")
    negative_prompt: Optional[str] = Field("worst quality, low quality, nsfw", description="Negative prompt")
    seed: Optional[int] = Field(None, ge=0, le=2**32-1, description="Random seed for reproducibility")
    webhook_url: Optional[str] = Field(None, description="URL to receive completion webhook")
    
    @validator('seed', pre=True, always=True)
    def set_seed(cls, v):
        if v is None:
            return int(time.time() * 1000) % (2**32)
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A beautiful cinematic shot of a sunset over the ocean",
                "negative_prompt": "worst quality, low quality",
                "seed": 42
            }
        }

class T2VRequest(BaseWorkflowRequest):
    """Text-to-Video request"""
    pass

class I2VRequest(BaseWorkflowRequest):
    """Image-to-Video request"""
    image_b64: str = Field(..., description="Source image encoded in Base64")
    
    @validator('image_b64')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoded image")

class S2VRequest(BaseWorkflowRequest):
    """Sound-to-Video request"""
    image_b64: str = Field(..., description="Reference image encoded in Base64")
    audio_b64: str = Field(..., description="Audio file encoded in Base64")
    
    @validator('image_b64', 'audio_b64')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoded data")

class CameraRequest(BaseWorkflowRequest):
    """Camera motion request"""
    image_b64: str = Field(..., description="Source image encoded in Base64")
    camera_motion: CameraMotion = Field(CameraMotion.ZOOM_IN, description="Camera motion type")
    
    @validator('image_b64')
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 encoded image")

class VideoResponse(BaseModel):
    """Response model for video generation"""
    status: TaskStatus
    video_b64: Optional[str] = None
    prompt_id: str
    processing_time: Optional[float] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "completed",
                "video_b64": "base64_encoded_video_data...",
                "prompt_id": "abc-123-def",
                "processing_time": 45.2
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    comfyui_status: str
    available_workflows: List[str]

# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================

def download_models():
    """Download all required models from HuggingFace"""
    from huggingface_hub import hf_hub_download
    
    base_model_path = Path("/root/comfy/ComfyUI/models")
    stats = {"total": 0, "success": 0, "failed": 0}
    
    logger.info("="*80)
    logger.info("Starting model download...")
    logger.info("="*80)
    
    for model_type, models in MODEL_REGISTRY.items():
        target_dir = base_model_path / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {model_type}...")
        
        for filename, details in models.items():
            stats["total"] += 1
            target_path = target_dir / filename
            
            try:
                if target_path.exists() and target_path.is_symlink():
                    logger.info(f"  ✓ {filename} already exists")
                    stats["success"] += 1
                    continue
                
                logger.info(f"  ⬇ Downloading {filename}...")
                cached_path = hf_hub_download(
                    repo_id=details["repo_id"],
                    filename=details["filename"],
                    cache_dir="/cache"
                )
                
                # Create symlink
                if target_path.exists():
                    target_path.unlink()
                
                os.symlink(cached_path, target_path)
                logger.info(f"  ✓ {filename} downloaded and linked")
                stats["success"] += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed to download {filename}: {e}")
                stats["failed"] += 1
    
    logger.info("="*80)
    logger.info(f"Model download complete: {stats['success']}/{stats['total']} successful, {stats['failed']} failed")
    logger.info("="*80)
    
    if stats["failed"] > 0:
        raise RuntimeError(f"Failed to download {stats['failed']} models")

def download_workflows():
    """Download all workflow JSON files"""
    workflow_dir = Path("/root/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading workflow files...")
    
    for name, url in WORKFLOW_URLS.items():
        filepath = workflow_dir / f"{name}.json"
        try:
            subprocess.run(
                ["wget", "-O", str(filepath), url],
                check=True,
                capture_output=True
            )
            logger.info(f"  ✓ Downloaded {name}.json")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Failed to download {name}.json: {e}")
            raise

# ============================================================================
# 6. MODAL IMAGE SETUP
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "comfy-cli==1.4.1",
        "fastapi[standard]",
        "uvicorn",
        "pydantic>=2.0.0",
        "requests",
        "websocket-client",
        "python-multipart",
        force_build=True
    )
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.54",
        force_build=True
    )
    .run_commands(
        "comfy node install ComfyUI-VideoHelperSuite",
        "comfy node install was-node-suite-comfyui",
        "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        force_build=True
    )
    .pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Setup volumes
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
input_vol = modal.Volume.from_name("comfy-input-vol", create_if_missing=True)

# Download models and workflows during image build
image = image.run_function(
    download_models,
    volumes={"/cache": cache_vol},
    timeout=4200
)

image = image.run_function(
    download_workflows,
    timeout=300
)

app = modal.App(name="comfyui-wan-production", image=image)

# ============================================================================
# 7. WORKFLOW PROCESSOR CLASS
# ============================================================================

class WorkflowProcessor:
    """Handles workflow loading and parameter injection"""
    
    def __init__(self, workflow_dir: Path):
        self.workflow_dir = workflow_dir
        self._workflow_cache = {}
    
    def load_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Load workflow JSON from file with caching"""
        if workflow_name not in self._workflow_cache:
            workflow_path = self.workflow_dir / f"{workflow_name}.json"
            
            if not workflow_path.exists():
                raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
            
            with open(workflow_path, 'r') as f:
                self._workflow_cache[workflow_name] = json.load(f)
            
            logger.info(f"Loaded workflow: {workflow_name}")
        
        # Return a deep copy to avoid mutation
        return json.loads(json.dumps(self._workflow_cache[workflow_name]))
    
    def inject_parameters(
        self,
        workflow: Dict[str, Any],
        mapping: WorkflowMapping,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject request parameters into workflow based on mapping"""
        
        for param_name, node_mapping in mapping.node_mappings.items():
            if param_name not in parameters:
                continue
            
            param_value = parameters[param_name]
            
            # Apply value to all mapped nodes
            for node_id, input_key in node_mapping.items():
                if node_id not in workflow:
                    logger.warning(f"Node {node_id} not found in workflow")
                    continue
                
                if "inputs" not in workflow[node_id]:
                    workflow[node_id]["inputs"] = {}
                
                workflow[node_id]["inputs"][input_key] = param_value
                logger.debug(f"Set {node_id}.inputs.{input_key} = {param_value}")
        
        return workflow

# ============================================================================
# 8. COMFYUI RUNNER CLASS
# ============================================================================

@app.cls(
    gpu="L40S",
    volumes={
        "/cache": cache_vol,
        "/root/comfy/ComfyUI/input": input_vol
    },
    container_idle_timeout=300,
    timeout=3600,
)
class ComfyRunner:
    """Main class for running ComfyUI workflows"""
    
    WORKFLOW_DIR = Path("/root/workflows")
    INPUT_DIR = Path("/root/comfy/ComfyUI/input")
    
    @modal.enter()
    def startup(self):
        """Initialize ComfyUI server"""
        self.client_id = str(uuid.uuid4())
        self.workflow_processor = WorkflowProcessor(self.WORKFLOW_DIR)
        
        logger.info(f"Starting ComfyUI server with client_id: {self.client_id}")
        
        # Start ComfyUI server
        self.proc = subprocess.Popen(
            f"comfy launch -- --listen {COMFYUI_HOST} --port {COMFYUI_PORT}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        for i in range(SERVER_STARTUP_TIMEOUT):
            try:
                response = requests.get(
                    f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/system_stats",
                    timeout=1
                )
                if response.status_code == 200:
                    logger.info(f"ComfyUI server ready after {i+1} seconds")
                    return
            except requests.ConnectionError:
                time.sleep(1)
        
        raise RuntimeError(f"ComfyUI server failed to start within {SERVER_STARTUP_TIMEOUT} seconds")
    
    def _save_file(self, b64_data: str, extension: str) -> str:
        """Save base64 data to input directory"""
        try:
            data = base64.b64decode(b64_data)
            filename = f"input_{uuid.uuid4()}.{extension}"
            filepath = self.INPUT_DIR / filename
            
            with open(filepath, "wb") as f:
                f.write(data)
            
            logger.info(f"Saved file: {filename} ({len(data)/1024:.2f} KB)")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise ValueError(f"Invalid base64 data: {e}")
    
    def _queue_workflow(self, workflow: Dict[str, Any]) -> str:
        """Submit workflow to ComfyUI"""
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            
            response = requests.post(
                f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/prompt",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "error" in result:
                raise RuntimeError(f"ComfyUI error: {result['error']}")
            
            prompt_id = result.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id in response: {result}")
            
            logger.info(f"Workflow queued with prompt_id: {prompt_id}")
            return prompt_id
            
        except requests.RequestException as e:
            logger.error(f"Failed to queue workflow: {e}")
            raise RuntimeError(f"Failed to submit workflow: {e}")
    
    def _poll_for_output(self, prompt_id: str) -> Optional[bytes]:
        """Poll for workflow completion and retrieve output"""
        start_time = time.time()
        
        for attempt in range(MAX_POLL_ATTEMPTS):
            try:
                # Check history
                response = requests.get(
                    f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/history/{prompt_id}",
                    timeout=10
                )
                response.raise_for_status()
                history = response.json()
                
                if prompt_id not in history:
                    logger.debug(f"Prompt {prompt_id} not in history yet (attempt {attempt+1})")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                prompt_history = history[prompt_id]
                
                # Check for errors
                if "status" in prompt_history:
                    status = prompt_history["status"]
                    if status.get("status_str") == "error":
                        error_msg = status.get("messages", ["Unknown error"])
                        raise RuntimeError(f"Workflow failed: {error_msg}")
                
                # Check for outputs
                outputs = prompt_history.get("outputs", {})
                if not outputs:
                    logger.debug(f"No outputs yet (attempt {attempt+1})")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                # Find video output
                for node_id, node_output in outputs.items():
                    if "videos" in node_output and node_output["videos"]:
                        video_info = node_output["videos"][0]
                        filename = video_info["filename"]
                        subfolder = video_info.get("subfolder", "")
                        file_type = video_info.get("type", "output")
                        
                        # Download video
                        params = {
                            "filename": filename,
                            "type": file_type
                        }
                        if subfolder:
                            params["subfolder"] = subfolder
                        
                        video_response = requests.get(
                            f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/view",
                            params=params,
                            timeout=60
                        )
                        video_response.raise_for_status()
                        
                        video_data = video_response.content
                        processing_time = time.time() - start_time
                        
                        logger.info(
                            f"Video retrieved: {len(video_data)/1024/1024:.2f} MB "
                            f"in {processing_time:.2f}s"
                        )
                        
                        return video_data
                
                logger.debug(f"No video output found yet (attempt {attempt+1})")
                time.sleep(POLL_INTERVAL)
                
            except requests.RequestException as e:
                logger.warning(f"Polling error (attempt {attempt+1}): {e}")
                time.sleep(POLL_INTERVAL)
        
        raise TimeoutError(
            f"Workflow did not complete within {MAX_POLL_ATTEMPTS * POLL_INTERVAL} seconds"
        )
    
    @modal.method()
    def generate_video(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        files: Dict[str, str] = None
    ) -> VideoResponse:
        """
        Generate video using specified workflow
        
        Args:
            workflow_name: Name of workflow to use (e.g., "t2v_14B")
            parameters: Dict of parameters to inject
            files: Dict of base64 encoded files {file_type: b64_data}
        """
        start_time = time.time()
        
        try:
            # Get workflow mapping
            if workflow_name not in WORKFLOW_MAPPINGS:
                raise ValueError(f"Unknown workflow: {workflow_name}")
            
            mapping = WORKFLOW_MAPPINGS[workflow_name]
            
            # Save files if provided
            if files:
                for file_type, b64_data in files.items():
                    extension = "png" if file_type == "image" else "mp3"
                    filename = self._save_file(b64_data, extension)
                    parameters[file_type] = filename
            
            # Load and inject parameters
            workflow = self.workflow_processor.load_workflow(mapping.workflow_file)
            workflow = self.workflow_processor.inject_parameters(
                workflow, mapping, parameters
            )
            
            # Queue workflow
            prompt_id = self._queue_workflow(workflow)
            
            # Poll for output
            video_data = self._poll_for_output(prompt_id)
            
            if not video_data:
                raise RuntimeError("No video output received")
            
            # Encode to base64
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            processing_time = time.time() - start_time
            
            return VideoResponse(
                status=TaskStatus.COMPLETED,
                video_b64=video_b64,
                prompt_id=prompt_id,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            processing_time = time.time() - start_time
            
            return VideoResponse(
                status=TaskStatus.FAILED,
                video_b64=None,
                prompt_id=str(uuid.uuid4()),
                processing_time=processing_time,
                error=str(e)
            )
    
    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Check health of ComfyUI server"""
        try:
            response = requests.get(
                f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/system_stats",
                timeout=5
            )
            return {
                "comfyui_status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_code": response.status_code
            }
        except Exception as e:
            return {
                "comfyui_status": "unhealthy",
                "error": str(e)
            }

# ============================================================================
# 9. FASTAPI APPLICATION
# ============================================================================

web_app = FastAPI(
    title="ComfyUI Wan 2.2 Video Generation API",
    description="Production-ready API for video generation using Wan 2.2 models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize runner
runner = ComfyRunner()

# ============================================================================
# 10. API ENDPOINTS
# ============================================================================

@web_app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "message": "ComfyUI Wan 2.2 Video Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        health_data = runner.health_check.remote()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            comfyui_status=health_data.get("comfyui_status", "unknown"),
            available_workflows=list(WORKFLOW_MAPPINGS.keys())
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@web_app.get("/workflows", response_model=Dict[str, Any])
async def list_workflows():
    """List all available workflows and their configurations"""
    workflows = {}
    for name, mapping in WORKFLOW_MAPPINGS.items():
        workflows[name] = {
            "workflow_file": mapping.workflow_file,
            "required_files": mapping.required_files,
            "parameters": list(mapping.node_mappings.keys())
        }
    return {"workflows": workflows}

@web_app.post("/text-to-video", response_model=VideoResponse, tags=["Video Generation"])
async def text_to_video(request: T2VRequest):
    """
    Generate video from text prompt using 14B model
    
    - Pure text-to-video generation
    - No input image required
    - High quality output with 14B parameter model
    """
    try:
        logger.info(f"T2V request: prompt='{request.prompt[:50]}...', seed={request.seed}")
        
        result = runner.generate_video.remote(
            workflow_name="t2v_14B",
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"T2V generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/image-to-video", response_model=VideoResponse, tags=["Video Generation"])
async def image_to_video(request: I2VRequest):
    """
    Generate video from image and text prompt using 14B model
    
    - Animate a static image
    - Text prompt guides the animation
    - 14B parameter model for high quality
    """
    try:
        logger.info(f"I2V request: prompt='{request.prompt[:50]}...', seed={request.seed}")
        
        result = runner.generate_video.remote(
            workflow_name="i2v_14B",
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed
            },
            files={"image": request.image_b64}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"I2V generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/sound-to-video", response_model=VideoResponse, tags=["Video Generation"])
async def sound_to_video(request: S2VRequest):
    """
    Generate video from reference image and audio using 14B model
    
    - Sync video to audio input
    - Reference image provides visual context
    - Audio-driven animation
    """
    try:
        logger.info(f"S2V request: prompt='{request.prompt[:50]}...', seed={request.seed}")
        
        result = runner.generate_video.remote(
            workflow_name="s2v_14B",
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed
            },
            files={
                "image": request.image_b64,
                "audio": request.audio_b64
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"S2V generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/camera-motion", response_model=VideoResponse, tags=["Video Generation"])
async def camera_motion(request: CameraRequest):
    """
    Generate video with camera motion effects using 14B model
    
    - Apply camera movements to static image
    - Support for zoom, pan, tilt, and static
    - Cinematic camera effects
    """
    try:
        logger.info(
            f"Camera request: prompt='{request.prompt[:50]}...', "
            f"motion={request.camera_motion}, seed={request.seed}"
        )
        
        result = runner.generate_video.remote(
            workflow_name="camera_14B",
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "camera_motion": request.camera_motion.value,
                "seed": request.seed
            },
            files={"image": request.image_b64}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Camera motion generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/text-image-to-video-5b", response_model=VideoResponse, tags=["Video Generation"])
async def text_image_to_video_5b(request: I2VRequest):
    """
    Generate video from text and image using 5B model
    
    - Faster generation with 5B model
    - Good quality with lower resource usage
    - Combines text and image input
    """
    try:
        logger.info(f"TI2V-5B request: prompt='{request.prompt[:50]}...', seed={request.seed}")
        
        result = runner.generate_video.remote(
            workflow_name="ti2v_5B",
            parameters={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "seed": request.seed
            },
            files={"image": request.image_b64}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"TI2V-5B generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/custom-workflow", response_model=VideoResponse, tags=["Advanced"])
async def custom_workflow(
    workflow_name: str = Body(..., description="Name of workflow to use"),
    parameters: Dict[str, Any] = Body(..., description="Parameters to inject"),
    files: Optional[Dict[str, str]] = Body(None, description="Base64 encoded files")
):
    """
    Advanced endpoint: Run any workflow with custom parameters
    
    - Specify workflow name
    - Provide custom parameters
    - Optional file inputs
    """
    try:
        if workflow_name not in WORKFLOW_MAPPINGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown workflow: {workflow_name}. Available: {list(WORKFLOW_MAPPINGS.keys())}"
            )
        
        logger.info(f"Custom workflow request: {workflow_name}")
        
        result = runner.generate_video.remote(
            workflow_name=workflow_name,
            parameters=parameters,
            files=files
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom workflow failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 11. ERROR HANDLERS
# ============================================================================

@web_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@web_app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# ============================================================================
# 12. MODAL DEPLOYMENT
# ============================================================================

@app.function()
@modal.asgi_app()
def serve():
    """Mount FastAPI application for Modal deployment"""
    return web_app

# ============================================================================
# 13. LOCAL TESTING
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
