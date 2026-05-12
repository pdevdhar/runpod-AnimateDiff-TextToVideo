import os

# set CUDA_MODULE_LOADING=LAZY to speed up the serverless function
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# set SAFETENSORS_FAST_GPU=1 to speed up the serverless function
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import time
import torch
import imageio
import tempfile
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

from animatediff.utils.util import init_pipeline, reload_motion_module, load_base_model, apply_lora, apply_motion_lora


def save_video(frames: torch.Tensor, seed=""):
    # save seed to the fil e name, for reproducibility
    output_video_path = tempfile.NamedTemporaryFile(prefix="{}_".format(seed), suffix=".mp4").name
    frames = (rearrange(frames, "b c t h w -> t b h w c").squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    writer = imageio.get_writer(output_video_path, fps=8, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_video_path


def check_data_format(job_input):
    # must have prompt in the input, otherwise raise error to the user
    if "prompt" in job_input:
        prompt = job_input["prompt"]
    else:
        raise ValueError("The input must contain a prompt.")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")

    # optional params, make sure they are in the right format here, otherwise raise error to the user
    steps          = job_input["steps"] if "steps" in job_input else None
    width          = job_input["width"] if "width" in job_input else None
    height         = job_input["height"] if "height" in job_input else None
    n_prompt       = job_input["n_prompt"] if "n_prompt" in job_input else None
    guidance_scale = job_input["guidance_scale"] if "guidance_scale" in job_input else None
    seed           = job_input["seed"] if "seed" in job_input else None
    base_model     = job_input["base_model"] if "base_model" in job_input else None
    base_loras     = job_input["base_loras"] if "base_loras" in job_input else None
    motion_lora    = job_input["motion_lora"] if "motion_lora" in job_input else None

    # check optional params
    if steps is not None and not isinstance(steps, int):
        raise ValueError("steps must be an integer.")
    if width is not None and not isinstance(width, int):
        raise ValueError("width must be an integer.")
    if height is not None and not isinstance(height, int):
        raise ValueError("height must be an integer.")
    if n_prompt is not None and not isinstance(n_prompt, str):
        raise ValueError("n_prompt must be a string.")
    if guidance_scale is not None and not isinstance(guidance_scale, float) and not isinstance(guidance_scale, int):
        raise ValueError("guidance_scale must be a float or an integer.")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if base_model is not None and not isinstance(base_model, str):
        raise ValueError("base_model must be a string.")
    if base_loras is not None:
        if not isinstance(base_loras, dict):
            raise ValueError("base_loras must be a dictionary.")
        for lora_name, lora_params in base_loras.items():
            if not isinstance(lora_name, str):
                raise ValueError("base_loras keys must be strings.")
            if not isinstance(lora_params, list):
                raise ValueError("base_loras values must be lists.")
            if len(lora_params) != 2:
                raise ValueError("base_loras values must be lists of length 2.")
            if not isinstance(lora_params[0], str):
                raise ValueError("base_loras values must be lists of strings.")
            if not isinstance(lora_params[1], float):
                raise ValueError("base_loras values must be lists of floats.")
    if motion_lora is not None:
        if not isinstance(motion_lora, list):
            raise ValueError("motion_lora must be a list.")
        if len(motion_lora) != 2:
            raise ValueError("motion_lora must be a list of length 2.")
        if not isinstance(motion_lora[0], str):
            raise ValueError("motion_lora must be a list of strings.")
        if (not isinstance(motion_lora[1], float)) and (not isinstance(motion_lora[1], int)):
            raise ValueError("motion_lora must be a list of floats.")
    return {
        "prompt"        : prompt,
        "steps"         : steps,
        "width"         : width,
        "height"        : height,
        "n_prompt"      : n_prompt,
        "guidance_scale": guidance_scale,
        "seed"          : seed,
        "base_model"    : base_model,
        "base_loras"    : base_loras,
        "motion_lora"   : motion_lora,
    }

import os
import torch

from diffusers import AnimateDiffPipeline, MotionAdapter, StableDiffusionPipeline


class AnimateDiff:
    def __init__(
        self,
        base_model="runwayml/stable-diffusion-v1-5",
        motion_adapter_repo="guoyww/animatediff-motion-adapter-v1-5-2",
        device="cuda"
    ):
        self.base_model = base_model
        self.motion_adapter_repo = motion_adapter_repo
        self.device = device

        self.pipe = None
        self._load_pipeline()

    # -----------------------------
    # PIPELINE LOADER
    # -----------------------------
    def _load_pipeline(self):

        print("[AnimateDiff] Loading base model...")

        # 1. Load motion adapter (SAFE, HF-based)
        try:
            adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_repo
            )
        except Exception as e:
            raise RuntimeError(
                f"[AnimateDiff] Failed to load motion adapter: {e}"
            )

        # 2. Build pipeline
        try:
            self.pipe = AnimateDiffPipeline.from_pretrained(
                self.base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16
            )
            self.pipe.to(self.device)

        except Exception as e:
            raise RuntimeError(
                f"[AnimateDiff] Pipeline load failed: {e}"
            )

        print("[AnimateDiff] Ready.")

    # -----------------------------
    # INFERENCE
    # -----------------------------
    def inference(
        self,
        prompt,
        steps=25,
        width=768,
        height=432,
        n_prompt="low quality, blurry",
        guidance_scale=7.5,
        seed=42,
        num_frames=16
    ):

        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=n_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_frames=num_frames,
            generator=generator
        )

        # save video
        import imageio
        import uuid

        path = f"/tmp/{uuid.uuid4()}.mp4"

        imageio.mimsave(path, result.frames[0], fps=6)

        return path

if __name__ == "__main__":
    # example seeds:
    # Person: 445608568
    # Default : 195577361
    import json

    animate_diff = AnimateDiff()

    # simple config
    with open("test_input_simple.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # faster config
    save_path = animate_diff.inference(prompt=test_input["prompt"])
    print("Result of simple config is saved to: {}\n".format(save_path))

    # complex custom config
    with open("./test_input.json", "r") as f:
        test_input = json.load(f)["input"]

    # only for testing
    test_input = check_data_format(test_input)

    # better config
    save_path = animate_diff.inference(
        prompt         = test_input["prompt"],
        steps          = test_input["steps"],
        width          = test_input["width"],
        height         = test_input["height"],
        n_prompt       = test_input["n_prompt"],
        guidance_scale = test_input["guidance_scale"],
        seed           = test_input["seed"],
        base_model     = test_input["base_model"],
        base_loras     = test_input["base_loras"],
        motion_lora    = test_input["motion_lora"],
    )
    print("Result of custom config is saved to: {}\n".format(save_path))
