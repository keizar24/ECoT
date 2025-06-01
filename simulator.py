#!/usr/bin/env python3
# simulator.py
# -----------------------------------------------------------
# End-to-end PyBullet + OpenVLA demo
# * updated so the visual-language model always receives a
#   **PIL.Image**, never a raw NumPy array                      <─── FIX
# * still uses 4-bit QLoRA (BitsAndBytes) for the 7-B model
# -----------------------------------------------------------

import os
import enum
import time
import textwrap
import cv2
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from urdf_models import models_data

# ────────────────────────────────────────────────────────────
# Model setup
# ────────────────────────────────────────────────────────────
MODEL_ID = "Embodied-CoT/ecot-openvla-7b-bridge"
DEVICE   = torch.device("cuda:0")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit          = True,
    bnb_4bit_quant_type   = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype    = torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config = bnb_cfg,
    torch_dtype         = torch.bfloat16,
    trust_remote_code   = True,
    device_map          = {"": 0},     # all weights on cuda:0
    low_cpu_mem_usage   = True,
)
model.eval()


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def build_prompt(instruction: str) -> str:
    return (
        f"{SYSTEM_PROMPT} "
        f"USER: What action should the robot take to {instruction.lower()}? "
        f"ASSISTANT: TASK:"
    )

INSTRUCTION      = "pick up the bottle"
TEXT_PROMPT      = build_prompt(INSTRUCTION)
MAX_NEW_TOKENS   = 1024                      # cut off for generation
CAM_W, CAM_H     = 640, 480


def load_object(path, position, orientation=None,
                rgbaColor=None, scaling=1.0, mass=0.1,
                use_cached_shapes=True, useFixedBase=False):
    """
    Generic loader for URDF, SDF, MJCF, Bullet snapshots *or* raw mesh files.
    Returns a list of body IDs or a single body ID.
    """
    if orientation is None:
        orientation = p.getQuaternionFromEuler([0, 0, 0])

    flags = 0
    if use_cached_shapes:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    ext = os.path.splitext(path)[1].lower()
    ids = []

    if ext == ".urdf":
        ids.append(p.loadURDF(
            path, position, orientation, globalScaling=scaling, flags=flags, useFixedBase=useFixedBase))
    elif ext in {".sdf", ".world"}:
        ids = p.loadSDF(path)
        for bid in ids:
            p.resetBasePositionAndOrientation(bid, position, orientation)
    elif ext == ".xml":
        ids = p.loadMJCF(path, flags=flags)
        for bid in ids:
            p.resetBasePositionAndOrientation(bid, position, orientation)
    elif ext == ".bullet":
        ids = p.loadBullet(
            path, basePosition=position, baseOrientation=orientation)
    elif ext in {".obj", ".stl", ".dae"}:
        col_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=path,
                                              meshScale=[scaling] * 3)
        vis_id = p.createVisualShape(p.GEOM_MESH,
                                           fileName=path,
                                           meshScale=[scaling] * 3,
                                           rgbaColor=rgbaColor or [1, 1, 1, 1])
        body_id = p.createMultiBody(baseMass=mass,
                                          baseCollisionShapeIndex=col_id,
                                          baseVisualShapeIndex=vis_id,
                                          basePosition=position,
                                          baseOrientation=orientation)
        ids.append(body_id)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if rgbaColor is not None and ext not in {".obj", ".stl", ".dae"}:
        for bid in ids:
            p.changeVisualShape(bid, -1, rgbaColor=rgbaColor)

    return ids if len(ids) > 1 else ids[0]


# ────────────────────────────────────────────────────────────
# PyBullet helpers
# ────────────────────────────────────────────────────────────
def setup_pybullet(gui: bool = True):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")

    ycb_model_path = "ycb_urdfs/ycb_assets/"

    obj_1_ori = p.getQuaternionFromEuler([0, 0, 90])
    obj_1_id = load_object(
        ycb_model_path + "026_sponge.urdf",
        np.array([0.1, 0.5, 0.05]),
        orientation=obj_1_ori,
        scaling=0.14
    )

    obj_3_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_3_id = load_object(
        ycb_model_path + "017_orange.urdf",
        np.array([0.04, 0.9, 0.05]),
        orientation=obj_3_ori,
        scaling=0.1
    )

    obj_4_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_4_id = load_object(
        models_data.model_lib()['yellow_bowl'],
        np.array([0.17, 0.8, 0.05]),
        orientation=obj_4_ori,
        scaling=1.3
    )

    obj_5_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_5_id = load_object(
        models_data.model_lib()['green_bowl'],
        np.array([0.04, 0.9, 0.05]),
        orientation=obj_5_ori,
        scaling=1.3
    )

    obj_6_ori = p.getQuaternionFromEuler([0, 0, 180])
    obj_6_id = load_object(
        models_data.model_lib()['bleach_cleanser'],
        np.array([0.4, 0.55, 0.11]),
        orientation=obj_6_ori
    )

    obj_7_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_7_id = load_object(
        ycb_model_path + "003_cracker_box.urdf",
        np.array([0.3, 0.95, 0.11]),
        orientation=obj_7_ori,
        scaling=0.09
    )

    obj_9_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_9_id = load_object(
        ycb_model_path + "065-d_cups.urdf",
        np.array([-0.20, 0.75, 0.11]),
        orientation=obj_9_ori,
        scaling=0.13
    )

    obj_11_ori = p.getQuaternionFromEuler([0, 0, 150])
    obj_11_id = load_object(
        models_data.model_lib()['clear_box'],
        np.array([-0.25, 0.8, 0.11]),
        orientation=obj_11_ori,
        scaling=1.5
    )

    # Simple camera pointing at the origin
    view_matrix  = p.computeViewMatrix(
        cameraEyePosition    = [1.2, 0.0, 0.8],
        cameraTargetPosition = [0.0, 0.0, 0.0],
        cameraUpVector       = [0.0, 0.0, 1.0]
    )
    proj_matrix  = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(CAM_W)/CAM_H,
        nearVal=0.01, farVal=10
    )

    franka_id = p.loadURDF("franka_panda/panda.urdf", [0,0,0],
                    useFixedBase=True)

    return dict(
        client   = physics_client,
        view_mat = view_matrix,
        proj_mat = proj_matrix,
        kuka     = franka_id,
    )


def get_camera_frame(view_mat, proj_mat):
    """Returns an RGB numpy array (H, W, 3) in **RGB** order."""
    w, h, rgba, depth, seg = p.getCameraImage(
        width        = CAM_W,
        height       = CAM_H,
        viewMatrix   = view_mat,
        projectionMatrix = proj_mat,
        renderer     = p.ER_BULLET_HARDWARE_OPENGL
    )
    rgba = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
    rgb  = rgba[..., :3]  # drop alpha
    return rgb


# ────────────────────────────────────────────────────────────
# Main control loop
# ────────────────────────────────────────────────────────────
def main():
    sim = setup_pybullet(gui=True)
    step_idx = 0

    try:
        while True:
            p.stepSimulation()

            # ── capture frame & convert to PIL  ─────────────────
            frame_rgb  = get_camera_frame(sim["view_mat"], sim["proj_mat"])
            pil_frame  = Image.fromarray(frame_rgb)      # <─── THIS is the fix

            # ── build model inputs  ─────────────────────────────
            inputs = processor(
                images=[pil_frame],          # batch of 1 image
                text=[TEXT_PROMPT],          # batch of 1 prompt
                return_tensors="pt",
            ).to(DEVICE, dtype=torch.bfloat16)

            # ── inference  ──────────────────────────────────────
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens   = MAX_NEW_TOKENS,
                    do_sample        = False,
                    pad_token_id     = processor.tokenizer.eos_token_id,
                )
            action_str = processor.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            )

            # ── display & (placeholder) execute  ───────────────
            print(f"[step {step_idx:04d}] VLM → {action_str}")
            # TODO: parse `action_str` → joint commands + execute

            step_idx += 1
            time.sleep(1.0 / 30.0)           # 30 Hz sim

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        p.disconnect()


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
