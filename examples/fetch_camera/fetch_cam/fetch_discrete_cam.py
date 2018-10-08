
from fetch_cam import FetchDiscreteEnv


# because thread bloack the image catch (maybe), so create the shell class 
class FetchDiscreteCamEnv:
    def __init__(self, dis_tolerance = 0.001, step_ds=0.005):
        self.env = FetchDiscreteEnv(dis_tolerance = 0.001, step_ds=0.005)


    def step(self,action):
        s, r, d, _ = self.env.step(action)

        # no use, but you need preserve it; otherwise, you will get error image
        rgb_external = self.env.sim.render(width=256, height=256, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
        rgb_gripper = self.env.sim.render(width=256, height=256, camera_name="gripper_camera_rgb", depth=False,
            mode='offscreen', device_id=-1)


        return rgb_gripper, r, d, None

    @property
    def pos(self):
        return self.env.pos

    @property
    def obj_pos(self):
        return self.env.obj_pos

    @property
    def gripper_state(self):
        return self.env.gripper_state

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

