from gym.envs.robotics.fetch_env import goal_distance

class FSM:
    _DIS_ERROR      = 0.01
    _PREGRIP_HEIGHT = 0.1
    fsm_state = ('idle', 'go_obj', 'down', 'grip', 'up', 'go_goal')

    # FSM(np.append(obs['eeinfo'][0], g), obs['achieved_goal'], goal, LIMIT_Z)
    def __init__(self, robot_state, obj_pos, goal_pos, limit_z=.415, step=60, skip_step=3):
        self.state = self.fsm_state[0]
        self.next_state = self.state
        # every task costs steps
        self._every_task = []
        self._step = 0
        self.maxstep = step
        self._done = False
        self.past_gs = None
        self._robot_state = None
        self.robot_state = robot_state
        self.obj_pos = obj_pos.copy()
        self.goal_pos = goal_pos.copy()
        self.limit_z = limit_z  # limit ee low height
        self.skip_step = skip_step

    @property
    def done(self):
        done, self._done = self._done, False
        return done

    @property
    def step(self):
        ''' Finished task spend step, Current task spend step '''
        return self._every_task, self._step

    @property
    def robot_state(self):
        return self._robot_state

    @robot_state.setter
    def robot_state(self, robot_state):
        assert robot_state.shape == (4,)
        if self._robot_state is not None:
            self.past_gs = self._robot_state[-1]
        self._robot_state = robot_state.copy()
        
    def execute(self):
        x, y, z, g = 0., 0., 0., 0

        print('self._step = ', self._step)
        print('self.state = ', self.state)

        if self.state == 'idle':
            self.next_state = 'go_obj'
            # output
            x, y, z, g = 0., 0., 0., 1.
        elif self.state == 'go_obj':
            self.next_state = 'down'
            # output
            x, y, z = self.obj_pos - self.robot_state[:3]
            z += self._PREGRIP_HEIGHT
            g = 1.
        elif self.state == 'down':
            self.next_state = 'grip'
            # output
            if self.obj_pos[2] <= self.limit_z:
                self.obj_pos[2] = self.limit_z
            x, y, z = self.obj_pos - self.robot_state[:3]
            g = 1.
        elif self.state == 'grip':
            self.next_state = 'up'
            # output
            x, y, z, g = 0., 0., 0., -1
        elif self.state == 'up':
            self.next_state = 'go_goal'
            # calculate target pos (up to object)
            self.tar_pos = self.obj_pos.copy()
            self.tar_pos[2] += self._PREGRIP_HEIGHT
            # output
            x, y, z = self.tar_pos - self.robot_state[:3]
            g = -1
        elif self.state == 'go_goal':
            # self.next_state = 'idle'
            # output
            x, y, z = self.goal_pos - self.robot_state[:3]
            g = -1
        
        if self._step > self.maxstep:
            self._done = True
            return x, y, z, g

        self._step += 1
        self.wait_robot()
        return x, y, z, g
            
    def wait_robot(self):
        if self.state == 'idle':
            if self._step < self.skip_step:
                return
        if self.state == 'go_obj':
            if goal_distance(self.robot_state[:2], self.obj_pos[:2]) > self._DIS_ERROR * 2:
                return
        elif self.state == 'down':
            if (goal_distance(self.robot_state[:2], self.obj_pos[:2]) > self._DIS_ERROR
                or self.robot_state[2] > self.obj_pos[2] + self._DIS_ERROR/2.0):
                return
        elif self.state == 'up':
            if (goal_distance(self.robot_state[:2], self.tar_pos[:2]) > self._DIS_ERROR * 2
                or self.robot_state[2] < self.tar_pos[2] - self._DIS_ERROR/2.0):
                return
        # Done!!!
        elif self.state == 'go_goal':
            if goal_distance(self.robot_state[:3], self.goal_pos) > self._DIS_ERROR * 3:
                return
            self._done = True
        elif self.state == 'grip':
            # print(self._step, self.past_gs, self.robot_state[-1])
            if (self._step < self.skip_step    or
                self.robot_state[-1] >= 0.05 or 
                self.past_gs - self.robot_state[-1] > self._DIS_ERROR/2.0):     
                return

        self.state = self.next_state
        self._every_task.append(self._step)
        self._step = 0