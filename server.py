import os
import base64
import io
import runpod
import torch
from PIL import Image
# FIX: Import SparseControlNetModel and the specific SparseControlNet pipeline directly
from diffusers import DDIMScheduler, MotionAdapter
from diffusers.models.controlnets.controlnet_sparsectrl import SparseControlNetModel
from diffusers.pipelines.animatediff.pipeline_animatediff_sparsectrl import AnimateDiffSparseControlNetPipeline
from diffusers.utils import export_to_video

# Preload models outside the handler loop for warm worker execution speed
device = "cuda" if torch.cuda.is_available() else "cpu"
volume_models_path = "/runpod-volume/models"

print("Initializing AnimateDiff + SparseCtrl Pipeline...")

# 1. Load the core AnimateDiff Motion Module from your volume
motion_adapter = MotionAdapter.from_pretrained(
    f"{volume_models_path}/Motion_Module", 
    torch_dtype=torch.float16
)

# 2. Load the newly downloaded SparseCtrl RGB condition model for Image-to-Video
sparsectrl_model = SparseControlNetModel.from_pretrained(
    f"{volume_models_path}/Motion_Module",
    subfolder="", 
    file_name="v3_sd15_sparsectrl_rgb.ckpt",
    torch_dtype=torch.float16
)

# 3. Instantiate the master pipeline with SparseCtrl support
# 3. FIX: Use the specific SparseControlNet pipeline constructor class
pipeline = AnimateDiffSparseControlNetPipeline.from_pretrained(
    f"{volume_models_path}/StableDiffusion", 
    motion_adapter=motion_adapter,
    controlnet=sparsectrl_model,
    torch_dtype=torch.float16
).to(device)

"""
pipeline = AnimateDiffPipeline.from_pretrained(
    f"{volume_models_path}/StableDiffusion", 
    motion_adapter=motion_adapter,
    controlnet=sparsectrl_model,
    torch_dtype=torch.float16
).to(device)
"""

# 4. Use DDIMScheduler for stable generation frame sequencing
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_vae_slicing()

print("Pipeline initialization complete. Waiting for jobs...")

def decode_base64_image(base64_string):
    """Decodes an incoming base64 string from your Flask app into a PIL Image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def image_to_video_handler(job):
    try:
        job_input = job.get("input", {})
        
        # Pull prompt settings safely
        prompt = job_input.get("prompt", "cinematic motion, high quality")
        n_prompt = job_input.get("n_prompt", "blurry, low quality, distorted")
        steps = int(job_input.get("steps", 20))
        guidance_scale = float(job_input.get("guidance_scale", 7.5))
        width = int(job_input.get("width", 512))
        height = int(job_input.get("height", 512))
        
        # Extract the optional base64 image field
        base64_image = job_input.get("image", None)
        
        if base64_image:
            print("Processing Image-to-Video request via SparseCtrl...")
            init_image = decode_base64_image(base64_image)
            init_image = init_image.resize((width, height))
            
            # Locks input image to frame 0 and animates the rest
            output = pipeline(
                prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                image=init_image, 
                num_frames=16
            )
        else:
            print("No image provided. Falling back to standard Text-to-Video...")
            output = pipeline(
                prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_frames=16
            )
            
        # Compile frames array into a web-compatible H.264 MP4 container
        frames = output.frames
        out_path = "/tmp/output_video.mp4"
        export_to_video(frames, out_path, fps=8)
        
        # Stream the MP4 binary out as a clean base64 string
        with open(out_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode("utf-8")
            
        return {"status": "success", "video_data": encoded_string}
        
    except Exception as e:
        print(f"CRITICAL HANDLER ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

# Boot the RunPod serverless listener loop
runpod.serverless.start({"handler": image_to_video_handler})

