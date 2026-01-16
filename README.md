# Wan 2.2 (A14B) RunPod Serverless API

Deploy Wan 2.2 video generation model on RunPod Serverless with 80GB VRAM GPU support. This setup uses GitHub Actions for Docker builds - no local Docker installation required.

## Features

- **Text-to-Video (T2V)**: Generate videos from text prompts
- **Image-to-Video (I2V)**: Animate images with text guidance
- **Configurable Parameters**: Resolution, frames, guidance scale, steps
- **Async Webhook Support**: For long-running generations

## Prerequisites

- GitHub account
- RunPod account with credits
- (Optional) Hugging Face account for gated models

## Quick Start

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub (e.g., `wan22-runpod`)
2. Clone this code and push to your repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/wan22-runpod.git
git push -u origin main
```

### Step 2: Configure GitHub Secrets (Optional)

If the model requires authentication, add these secrets to your repository:

1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Add `HF_TOKEN` with your Hugging Face token (if needed)

> Note: `GITHUB_TOKEN` is automatically provided by GitHub Actions for GHCR access.

### Step 3: Trigger Build

Push to the `main` branch to trigger the GitHub Actions workflow. The workflow will:

1. Build the Docker image
2. Push to GitHub Container Registry (GHCR)

Check the **Actions** tab to monitor build progress.

### Step 4: Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Custom** template
4. Configure:
   - **Container Image**: `ghcr.io/YOUR_USERNAME/wan22-runpod:latest`
   - **Container Disk**: 50 GB (for model caching)
   - **Volume Disk**: 100 GB (persistent model storage)
   - **Volume Mount Path**: `/runpod-volume`
5. Select GPU: **A100 80GB** or **H100 80GB**
6. Set worker configuration:
   - **Max Workers**: 1-3 (based on budget)
   - **Idle Timeout**: 60 seconds
   - **Execution Timeout**: 600 seconds
7. Click **Deploy**

### Step 5: Make Container Public (if needed)

If RunPod can't pull your image:

1. Go to GitHub > Your Profile > **Packages**
2. Find `wan22-runpod` package
3. Go to **Package settings**
4. Change visibility to **Public** or add RunPod as collaborator

## API Reference

### Endpoint URL

```
https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

For async requests:
```
https://api.runpod.ai/v2/{ENDPOINT_ID}/run
```

### Headers

```
Authorization: Bearer YOUR_RUNPOD_API_KEY
Content-Type: application/json
```

### Request Body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | string | Yes | `"t2v"` | Generation mode: `"t2v"` or `"i2v"` |
| `prompt` | string | Yes | - | Text description of the video |
| `negative_prompt` | string | No | `""` | What to avoid in generation |
| `image` | string | I2V only | - | Base64 encoded image for I2V |
| `num_frames` | int | No | `49` | Number of frames (17, 21, 25, ..., 81) |
| `resolution` | string | No | `"480p"` | Output resolution: `"480p"` or `"720p"` |
| `guidance_scale` | float | No | `5.0` | CFG scale (higher = more prompt adherence) |
| `num_inference_steps` | int | No | `30` | Denoising steps (more = better quality) |
| `seed` | int | No | random | Seed for reproducibility |

### Example: Text-to-Video

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "t2v",
      "prompt": "A serene lake surrounded by mountains at sunset, cinematic lighting, 4K quality",
      "negative_prompt": "blurry, low quality, distorted",
      "num_frames": 49,
      "resolution": "720p",
      "guidance_scale": 6.0,
      "num_inference_steps": 30
    }
  }'
```

### Example: Image-to-Video

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "i2v",
      "prompt": "The woman slowly turns her head and smiles",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "num_frames": 49,
      "resolution": "480p",
      "guidance_scale": 5.0
    }
  }'
```

### Example: Async with Webhook

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "t2v",
      "prompt": "A rocket launching into space with dramatic clouds",
      "num_frames": 81,
      "resolution": "720p"
    },
    "webhook": "https://your-server.com/webhook/callback"
  }'
```

### Response

```json
{
  "delayTime": 2500,
  "executionTime": 180000,
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "output": {
    "video": "AAAAIGZ0eXBpc29t...",
    "seed": 1234567890,
    "mode": "t2v",
    "resolution": "720p",
    "num_frames": 49,
    "generation_time_seconds": 175.23
  },
  "status": "COMPLETED"
}
```

The `video` field contains a base64-encoded MP4 file. Decode it to get your video:

```python
import base64

video_base64 = response["output"]["video"]
video_bytes = base64.b64decode(video_base64)

with open("output.mp4", "wb") as f:
    f.write(video_bytes)
```

## Performance Notes

| Metric | Value |
|--------|-------|
| Cold Start | ~2-3 minutes (model loading) |
| 49 frames @ 480p | ~2 minutes |
| 49 frames @ 720p | ~3-4 minutes |
| 81 frames @ 720p | ~5-6 minutes |
| GPU Cost (A100) | ~$0.0014/second |

## Troubleshooting

### Build Fails - Out of Disk Space

The GitHub Actions runner has limited disk space. The workflow includes a cleanup step, but if it still fails:

1. Don't bake models into the image (keep the model download lines commented in Dockerfile)
2. Models will download on first RunPod cold start instead

### RunPod Can't Pull Image

1. Make sure the GitHub Actions workflow completed successfully
2. Check if the package is public or if RunPod has access
3. Verify the image name matches exactly

### Out of Memory Errors

- Use A100 80GB or H100 80GB GPUs
- Reduce `num_frames` or use `480p` resolution
- Lower `num_inference_steps` (minimum 20 for decent quality)

### Slow Cold Starts

First request downloads ~50GB of model weights. To speed up:

1. Use RunPod volume storage (models cached between calls)
2. Set idle timeout to keep workers warm
3. Consider baking models into image (increases build time significantly)

## License

This project is for educational purposes. Wan 2.2 model usage is subject to the model's license terms.

