import mujoco
import mujoco.viewer
import numpy as np
# from utils import default_angles_config
from deploy_mujoco import default_angles_config

def load_model():
    # Path to your H1 MJCF
    model = mujoco.MjModel.from_xml_path("./assets/h1_description/mjcf/scene.xml")

    # ======= Basic physics setup =======
    model.opt.timestep = 0.001            # small, stable integration step
    # model.opt.gravity[:] = [0, 0, -9.81]  # normal Earth gravity
    model.opt.gravity[:] = [0, 0, 0]  # zero gravity for testing
    model.opt.iterations = 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # Euler or RK4
    
    # Add damping to prevent jitter / free base blowup
    model.dof_damping[:] = 0.2

    # ======= Initialize data =======
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)      # clean reset
    mujoco.mj_forward(model, data)

    # ======= Ensure valid base pose =======
    # qpos[:3] = base position (x, y, z)
    data.qpos[:3] = np.array([0, 0, 1.5])   # start 1m above ground
    # qpos[3:7] = base quaternion (w, x, y, z)
    data.qpos[3:7] = np.array([1, 0, 0, 0]) # identity orientation

    mujoco.mj_forward(model, data)  # recompute transforms

    # ======= Zero control inputs =======
    data.ctrl[:] = default_angles_config.copy()
    print(data.qpos)

    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Initial gravity: {model.opt.gravity}")
    print(f"Initial base qpos: {data.qpos[:7]}")

    print("Joints (index -> name):")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {name}")

    print("Actuators (index -> name):")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")


    return model, data


def main():
    model, data = load_model()

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            # Step the physics
            mujoco.mj_step(model, data)

            # Optionally freeze control for debugging
            data.ctrl[:] = default_angles_config.copy()
            # print(data.qpos)
            # print(f"Base linvel = {data.qvel[3:6]}, Base angvel = {data.qvel[0:3]}")

            # Sync viewer
            v.sync()


if __name__ == "__main__":
    main()
