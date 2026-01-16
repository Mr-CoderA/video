"""
RunPod Serverless Handler for Wan 2.2 (A14B)
Supports both Text-to-Video (T2V) and Image-to-Video (I2V) generation.
"""

import os
import base64
import tempfile
import time
from io import BytesIO
from typing import Any

import runpod
import torch
import imageio
from PIL import Image
from huggingface_hub import login
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

# Login to Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    print(f"Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("Hugging Face login successful!")
else:
    print("WARNING: HF_TOKEN not found in environment variables!")

# Global model references
t2v_pipeline = None
i2v_pipeline = None

# Model configuration
T2V_MODEL_ID = "Wan-AI/Wan2.2-T2V-14B-Diffusers"
I2V_MODEL_ID = "Wan-AI/Wan2.2-I2V-14B-480P-Diffusers"

# Resolution presets
RESOLUTION_PRESETS = {
    "480p": (848, 480),
    "720p": (1280, 720),
}


def load_models() -> None:
    """Load Wan 2.2 T2V and I2V pipelines into VRAM."""
    global t2v_pipeline, i2v_pipeline

    print("Loading Wan 2.2 T2V pipeline...")
    vae_t2v = AutoencoderKLWan.from_pretrained(
        T2V_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=HF_TOKEN,
    )
    t2v_pipeline = WanPipeline.from_pretrained(
        T2V_MODEL_ID,
        vae=vae_t2v,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    t2v_pipeline.to("cuda")
    print("T2V pipeline loaded successfully.")

    print("Loading Wan 2.2 I2V pipeline...")
    image_encoder = CLIPVisionModel.from_pretrained(
        I2V_MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
        token=HF_TOKEN,
    )
    vae_i2v = AutoencoderKLWan.from_pretrained(
        I2V_MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        token=HF_TOKEN,
    )
    i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
        I2V_MODEL_ID,
        vae=vae_i2v,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    i2v_pipeline.to("cuda")
    print("I2V pipeline loaded successfully.")

    print("All models loaded and ready!")


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64 encoded image string to PIL Image."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def encode_video_to_base64(video_path: str) -> str:
    """Encode a video file to base64 string."""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def generate_t2v(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> str:
    """Generate video from text prompt using T2V pipeline."""
    global t2v_pipeline

    generator = torch.Generator(device="cuda").manual_seed(seed)

    output = t2v_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        video_path = tmp_file.name
        export_to_video(output.frames[0], video_path, fps=16)

    video_base64 = encode_video_to_base64(video_path)
    os.unlink(video_path)

    return video_base64


def generate_i2v(
    prompt: str,
    negative_prompt: str,
    image: Image.Image,
    width: int,
    height: int,
    num_frames: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
) -> str:
    """Generate video from image using I2V pipeline."""
    global i2v_pipeline

    image = image.resize((width, height))
    generator = torch.Generator(device="cuda").manual_seed(seed)

    output = i2v_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        video_path = tmp_file.name
        export_to_video(output.frames[0], video_path, fps=16)

    video_base64 = encode_video_to_base64(video_path)
    os.unlink(video_path)

    return video_base64


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Input parameters:
        - mode: "t2v" or "i2v" (required)
        - prompt: Text description (required)
        - negative_prompt: Negative prompt (optional)
        - image: Base64 encoded image for I2V mode (required for i2v)
        - num_frames: Number of frames 16-81 (default: 49)
        - resolution: "480p" or "720p" (default: "480p")
        - guidance_scale: CFG value (default: 5.0)
        - num_inference_steps: Denoising steps (default: 30)
        - seed: Random seed (optional, random if not provided)
    """
    job_input = job.get("input", {})

    # Extract parameters
    mode = job_input.get("mode", "t2v").lower()
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    image_base64 = job_input.get("image")
    num_frames = job_input.get("num_frames", 49)
    resolution = job_input.get("resolution", "480p")
    guidance_scale = job_input.get("guidance_scale", 5.0)
    num_inference_steps = job_input.get("num_inference_steps", 30)
    seed = job_input.get("seed", int(time.time()) % 2**32)

    # Validate mode
    if mode not in ["t2v", "i2v"]:
        return {"error": f"Invalid mode: {mode}. Must be 't2v' or 'i2v'."}

    # Validate prompt
    if not prompt:
        return {"error": "Prompt is required."}

    # Validate I2V requirements
    if mode == "i2v" and not image_base64:
        return {"error": "Image is required for I2V mode."}

    # Get resolution
    if resolution not in RESOLUTION_PRESETS:
        return {"error": f"Invalid resolution: {resolution}. Must be '480p' or '720p'."}
    
    width, height = RESOLUTION_PRESETS[resolution]

    # Validate num_frames (must be 4k+1 for Wan)
    valid_frames = [17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81]
    if num_frames not in valid_frames:
        closest = min(valid_frames, key=lambda x: abs(x - num_frames))
        num_frames = closest

    try:
        start_time = time.time()

        if mode == "t2v":
            video_base64 = generate_t2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        else:
            image = decode_base64_image(image_base64)
            video_base64 = generate_i2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

        generation_time = time.time() - start_time

        return {
            "video": video_base64,
            "seed": seed,
            "mode": mode,
            "resolution": resolution,
            "num_frames": num_frames,
            "generation_time_seconds": round(generation_time, 2),
        }

    except Exception as e:
        return {"error": str(e)}


# Load models on cold start
load_models()

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})

