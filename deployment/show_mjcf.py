import mujoco
import rerun as rr
import rerun_loader_mjcf

model = mujoco.MjModel.from_xml_path("./assets/h1_description/mjcf/h1.xml")
data = mujoco.MjData(model)

rr.init("mjcf_viewer", spawn=True)
logger = rerun_loader_mjcf.MJCFLogger(model)

rr.set_time("frame", sequence=0)
logger.log_model()

data.qpos[0] += 0.5
mujoco.mj_forward(model, data)

rr.set_time("frame", sequence=1)
logger.log_data(data)