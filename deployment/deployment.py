from deploy_sim import *


# def load_callback(model=None, data=None):
#     mujoco.set_mjcb_control(None)


#     model = mujoco.MjModel.from_xml_path('./h1_description/mjcf/scene.xml')
#     # model.opt.gravity[:] = [0, 0, 0]
#     model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
#     data = mujoco.MjData(model)


#     mujoco.mj_resetDataKeyframe(model, data, 1)

#     # ctrl_dt = 0.02
#     sim_dt = 0.001
#     n_substeps = 4
#     model.opt.timestep = sim_dt

#     policy = TorchController(
#         policy_path=POLICY_PATH,
#         default_angles=np.array(default_angles_config),
#         n_substeps=n_substeps,
#         action_scale=0.7,
#         vel_scale_x=0.5,
#         vel_scale_y=0.5,
#         vel_scale_rot=1.0,
#     )

#     mujoco.set_mjcb_control(policy.get_control)
#     # print(data.qpos)

#     return model, data


# if __name__ == "__main__":
#     viewer.launch(loader=load_callback)


def load_model():
    # Path to your H1 MJCF
    model = mujoco.MjModel.from_xml_path("./h1_description/mjcf/scene.xml")

    # ======= Basic physics setup =======
    model.opt.timestep = 0.005            # small, stable integration step
    # model.opt.gravity[:] = [0, 0, -9.81]  # normal Earth gravity
    # model.opt.gravity[:] = [0, 0, 0]  # zero gravity for testing

    model.opt.iterations = 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # Euler or RK4
    
    # Add damping to prevent jitter / free base blowup
    # model.dof_damping[:] = 0.2

    # ======= Initialize data =======
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)      # clean reset
    mujoco.mj_forward(model, data)

    # ======= Ensure valid base pose =======
    # qpos[:3] = base position (x, y, z)
    data.qpos[:3] = np.array([0, 0, 1.04])   # start 1m above ground
    # qpos[3:7] = base quaternion (w, x, y, z)
    data.qpos[3:7] = np.array([1, 0, 0, 0]) # identity orientation
    # data.qpos[3:7] = np.array([0.7071, 0, 0, -0.7071]) # 90 deg rotation around z-axis

    data.qpos[7:] = default_angles_config.copy()

    mujoco.mj_forward(model, data)  # recompute transforms

    # print(default_angles_config.copy())

    data.ctrl[:] = default_angles_config.copy()
    print(data.qpos)
    print(f"Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    print(f"Initial gravity: {model.opt.gravity}")
    # data.ctrl[:] = np.zeros_like(default_angles_config)
    # print(data.qpos)
    

    return model, data


def loaded():
    model, data = load_model()

    policy = TorchController(
        policy_path=POLICY_PATH,
        default_angles=np.array(default_angles_config),
        n_substeps=1,
        action_scale=0.5,
        vel_scale_x=1.0,
        vel_scale_y=1.0,
        vel_scale_rot=1.0,
    )

    # with mujoco.viewer.launch_passive(model, data) as v:
    #     while v.is_running():
    #         # Step the physics
    #         mujoco.mj_step(model, data)

    #         # Optionally freeze control for debugging
    #         data.ctrl[:] = policy.get_control(model, data)
    #         # print(data.qpos)

    #         # Sync viewer
    #         v.sync()

    mujoco.set_mjcb_control(policy.get_control)
    # viewer.launch(model, data)
    return model, data

if __name__ == "__main__":
    viewer.launch(loader=loaded)