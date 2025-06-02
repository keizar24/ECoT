#!/usr/bin/env python3
# simulator.py
# -----------------------------------------------------------
# End-to-end PyBullet + OpenVLA demo              2025-05
#
#  ▸ Loads the 7-B "ecot-openvla-bridge" model in 4-bit QLoRA.
#  ▸ Feeds the live PyBullet camera frame + task prompt to the
#    VLM at ~30 Hz.
#  ▸ `vlm.predict_action` returns a continuous 7-DoF delta that
#    is sent to the Franka Panda through one-shot IK + gripper.
# -----------------------------------------------------------

import time
import numpy as np
from PIL import Image
import pybullet as p
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from scipy.spatial.transform import Rotation

# local helpers you already have
from utils import setup_pybullet, get_camera_frame

# ────────────────────────────────────────────────────────────
# Model initialisation
# ────────────────────────────────────────────────────────────
MODEL_ID = "Embodied-CoT/ecot-openvla-7b-bridge"
DEVICE = torch.device("cuda:0")

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
vlm = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
vlm.eval()  # turn off dropout, etc.

# ────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
TASK_TEXT = "pick up the bottle"


def build_prompt(task: str) -> str:
    return (f"{SYSTEM_PROMPT} USER: What action should the robot take to {task.lower()}? "
            f"ASSISTANT: TASK:")


TEXT_PROMPT = build_prompt(TASK_TEXT)

# ⭐ FIX 6: Cache tokenized prompt to avoid re-tokenization every frame
CACHED_PROMPT_IDS = processor.tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"].to(DEVICE)

# Note: FIX 1 (history token) is commented out - this specific model doesn't support the past_action_ids parameter

# ────────────────────────────────────────────────────────────
# Runtime parameters
# ────────────────────────────────────────────────────────────
MAX_FPS = 30
IK_MAX_ITERS = 200
ARM_JOINTS = list(range(7))  # Panda joints q0…q6
GRIPPER_JOINTS = (9, 10)  # finger joints

# camera parameters you already use
EYE = np.array([0.075, -1.5, 0.8])
TARGET = np.array([0.075, -0.725, 0.08])
CAM_UP = np.array([0.0, 0.0, 1.0])

# build an orthonormal basis
forward_cam = TARGET - EYE
forward_cam /= np.linalg.norm(forward_cam)

# ⭐ FIX 2: Correct camera-to-world transform (up × forward, not forward × up)
right_cam = np.cross(CAM_UP, forward_cam)
right_cam /= np.linalg.norm(right_cam)

up_cam = np.cross(right_cam, forward_cam)  # already normalised

# 3×3 rotation matrix  (world_R_cam)
R_wc = np.column_stack((right_cam, up_cam, forward_cam))

# Ensure right-handed coordinate system (positive determinant)
if np.linalg.det(R_wc) < 0:
    # Flip the right vector to make it right-handed
    right_cam = -right_cam
    R_wc = np.column_stack((right_cam, up_cam, forward_cam))

# Convert to scipy Rotation for quaternion operations (FIX 3)
R_wc_scipy = Rotation.from_matrix(R_wc)


# -----------------------------------------------------------

# ────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────
def main() -> None:
    sim = setup_pybullet(gui=True)

    # ⭐ FIX 4: Set physics timestep to match render rate
    p.setTimeStep(1.0 / MAX_FPS)  # 30 Hz physics to match rendering

    step = 0

    try:
        while True:
            # 1) physics step
            p.stepSimulation()

            # 2) camera frame → PIL.Image
            rgb_np = get_camera_frame(sim["view_mat"], sim["proj_mat"])
            rgb_pil = Image.fromarray(rgb_np)

            # 3) tokenizer + vision pre-proc with cached prompt
            # ⭐ FIX 6: Use cached prompt IDs
            pixel_values = processor.image_processor(rgb_pil, return_tensors="pt")["pixel_values"].to(DEVICE,
                                                                                                      dtype=torch.bfloat16)

            inputs = {
                "input_ids": CACHED_PROMPT_IDS,
                "pixel_values": pixel_values,
            }

            # 4) VLM → continuous action (7-DoF) in metres/radians
            # ⭐ FIX 7: Add error handling for CUDA OOM
            try:
                with torch.no_grad():
                    action, generated_ids = vlm.predict_action(
                        **inputs,
                        unnorm_key="bridge_orig",  # match training scale
                        do_sample=False,
                        max_new_tokens=1,  # one action per tick
                    )
                # Note: FIX 1 (history token) disabled - this model doesn't support past_action_ids parameter

            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM error: {e}")
                print("Skipping this frame and continuing...")
                step += 1
                time.sleep(1.0 / MAX_FPS)
                continue

            # action is a tensor([dx, dy, dz, droll, dpitch, dyaw, dgrip])
            dx, dy, dz, droll, dpitch, dyaw, dgrip = action.tolist()

            # ---------------------------------------------
            # 5) current EE pose in WORLD frame
            link_state = p.getLinkState(sim["kuka"], 11, computeForwardKinematics=True)
            cur_xyz = np.array(link_state[4])
            cur_quat = np.array(link_state[5])

            # ---------------------------------------------
            # 6) map Δ from CAMERA to WORLD
            d_cam = np.array([dx, -dy, dz])  # (3,)
            d_world = R_wc @ d_cam  # (3,)

            tgt_xyz = cur_xyz + d_world

            # ⭐ FIX 3: Proper quaternion-based rotation instead of Euler approximation
            # Convert camera-frame delta Euler to quaternion
            d_rpy_cam = np.array([droll, dpitch, dyaw])
            d_quat_cam = Rotation.from_euler('XYZ', d_rpy_cam)

            # Transform delta quaternion to world frame
            d_quat_world = R_wc_scipy * d_quat_cam

            # Apply delta rotation to current orientation
            cur_rot = Rotation.from_quat(cur_quat)  # pybullet uses [x,y,z,w]
            tgt_rot = d_quat_world * cur_rot
            tgt_quat = tgt_rot.as_quat()  # returns [x,y,z,w] format for pybullet

            # ---------------------------------------------
            # 7) inverse kinematics, identical to before
            ik = p.calculateInverseKinematics(
                sim["kuka"], 11, tgt_xyz, tgt_quat,
                maxNumIterations=IK_MAX_ITERS, residualThreshold=1e-4)

            for j, q in zip(ARM_JOINTS, ik[:7]):
                p.setJointMotorControl2(sim["kuka"], j,
                                        p.POSITION_CONTROL,
                                        targetPosition=q,
                                        force=250,
                                        positionGain=0.1,
                                        velocityGain=1.0)

            # ---------------------------------------------
            # ⭐ FIX 5: Proportional gripper control instead of binary
            grip_pos = 0.04 * (np.clip(dgrip, -1, 1) + 1) / 2  # map [-1,1] to [0, 0.04]
            for j in GRIPPER_JOINTS:
                p.setJointMotorControl2(sim["kuka"], j,
                                        p.POSITION_CONTROL,
                                        targetPosition=grip_pos,
                                        force=50)

            # 9) console log
            print(f"[{step:04d}] Δxyz=({dx:+.3f},{dy:+.3f},{dz:+.3f})  "
                  f"Δrpy=({np.degrees([droll, dpitch, dyaw])})  "
                  f"grip={grip_pos:.3f}")

            step += 1
            time.sleep(1.0 / MAX_FPS)

    except KeyboardInterrupt:
        print("\nUser interrupted – shutting down.")
    finally:
        p.disconnect()


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()