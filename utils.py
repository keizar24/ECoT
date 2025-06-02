import pybullet as p
import pybullet_data
import numpy as np
import os
from urdf_models import models_data




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

CAM_W, CAM_H     = 640, 480

def setup_pybullet(gui: bool = True):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")

    ycb_model_path = "ycb_urdfs/ycb_assets/"

    obj_1_ori = p.getQuaternionFromEuler([0, 0, 90])
    obj_1_id = load_object(
        ycb_model_path + "026_sponge.urdf",
        np.array([0.1, -0.5, 0.05]),
        orientation=obj_1_ori,
        scaling=0.14
    )

    obj_3_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_3_id = load_object(
        ycb_model_path + "017_orange.urdf",
        np.array([0.04, -0.9, 0.05]),
        orientation=obj_3_ori,
        scaling=0.1
    )

    obj_4_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_4_id = load_object(
        models_data.model_lib()['yellow_bowl'],
        np.array([0.17, -0.8, 0.05]),
        orientation=obj_4_ori,
        scaling=1.3
    )

    obj_5_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_5_id = load_object(
        models_data.model_lib()['green_bowl'],
        np.array([0.04, -0.9, 0.05]),
        orientation=obj_5_ori,
        scaling=1.3
    )

    obj_6_ori = p.getQuaternionFromEuler([0, 0, 180])
    obj_6_id = load_object(
        models_data.model_lib()['bleach_cleanser'],
        np.array([0.4, -0.55, 0.11]),
        orientation=obj_6_ori
    )

    obj_7_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_7_id = load_object(
        ycb_model_path + "003_cracker_box.urdf",
        np.array([0.3, -0.95, 0.11]),
        orientation=obj_7_ori,
        scaling=0.09
    )

    obj_9_ori = p.getQuaternionFromEuler([0, 0, 0])
    obj_9_id = load_object(
        ycb_model_path + "065-d_cups.urdf",
        np.array([-0.20, -0.75, 0.11]),
        orientation=obj_9_ori,
        scaling=0.13
    )

    obj_11_ori = p.getQuaternionFromEuler([0, 0, 150])
    obj_11_id = load_object(
        models_data.model_lib()['clear_box'],
        np.array([-0.25, -0.8, 0.11]),
        orientation=obj_11_ori,
        scaling=1.5
    )

    # Camera positioned to cover all objects in scene
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.2, -0.725, 0.8],
        cameraTargetPosition=[0.075, -0.725, 0.08],
        cameraUpVector=[0.0, 0.0, 1.0]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(CAM_W) / CAM_H,
        nearVal=0.01, farVal=10
    )

    # Debug Camera Configuration
    camera_distance = 0.3
    camera_yaw = 0
    camera_pitch = -28
    camera_target_position = [0, -1.3, 0.6]
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    franka_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0],
                           useFixedBase=True)

    # Set initial robot position to resting joint positions (180 degree spun on XY plane)
    resting_jointPositions = [1.35 + 3.14159, 0.1, 0.31, -2.2, 0.0, 2.3, 2.967, 0.00, 0.00]
    for i in range(len(resting_jointPositions)):
        p.resetJointState(franka_id, i, resting_jointPositions[i])

    return dict(
        client=physics_client,
        view_mat=view_matrix,
        proj_mat=proj_matrix,
        kuka=franka_id,
    )


def get_camera_frame(view_mat, proj_mat):
    """Returns an RGB numpy array (H, W, 3) in **RGB** order."""
    w, h, rgba, depth, seg = p.getCameraImage(
        width=CAM_W,
        height=CAM_H,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgba = np.reshape(rgba, (h, w, 4)).astype(np.uint8)
    rgb = rgba[..., :3]  # drop alpha
    return rgb
