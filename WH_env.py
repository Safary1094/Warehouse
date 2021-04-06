from Entities import Agent, Boxes, Shelves, Stat, get_coord, get_ang, get_velo
from Plan import Plan
from Config import Config
import pybullet as pb
import matplotlib.pyplot as plt
import numpy as np
import math


class WH_env:
    def __init__(self, conf: Config):
        self.conf = conf
        self.stat = Stat()
        self.obst_map = np.zeros((self.conf.size[1], self.conf.size[1]))
        # self.agents = [Agent(urdf, self.conf, self.stat) for urdf in [self.conf.lift_red_path, self.conf.move_blue_path, self.conf.move_yellow_path]]
        self.agents = [Agent(urdf, self.conf, self.stat) for urdf in [self.conf.lift_red_path]]
        self.shelves = Shelves(self.conf)
        self.boxes = Boxes(self.conf, self.obst_map)
        self.plan = Plan(self.agents, self.boxes, self.shelves, self.conf)

        self.state = None

        self.done = None

    def plot_states(self, state):
        f = plt.figure(2)
        plt.clf()
        plt.subplots_adjust(top=0.99, bottom=0.025, left=0.015, right=1.0, hspace=0.12, wspace=0.12)
        plot_grid_size = (3, 10)
        pi = 0
        for i, s in enumerate(state):
            if i in [0, 2, 4, 6, 7]:
                continue
            for pix_map in s:
                plt.subplot(*plot_grid_size, pi + 1)
                plt.imshow(pix_map)
                pi += 1

        plt.pause(0.0001)
        plt.show()
        a = 1

    def reset_agents(self):
        self.obst_map.fill(0)
        for agent in self.agents:
            agent.id = None
            while agent.collision():
                agent.reset()
                pb.stepSimulation()

    def update_plan(self):
        self.stat.change_plan = False

        self.plan.initial_graph()
        self.plan.expand_plan()
        new_plan = self.plan.plans_list[0][-1]

        for agent in self.agents:
            agent.plan = new_plan.acts[agent.id][0][0]

        # self.draw_plan(new_plan)

    def draw_plan(self, new_plan):
        colors = [(1, 0, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5)]
        for color, agent in zip(colors, self.agents):
            # erase old plan
            for plan_id in agent.plan_mark_id:
                pb.removeBody(plan_id)
            agent.plan_mark_id.clear()

            # draw new plan
            for i, p in enumerate(new_plan.acts[agent.id]):
                target = pb.getBasePositionAndOrientation(p[0])[0]
                vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=0.2, length=2. * (i + 1), rgbaColor=color)
                agent.plan_mark_id.append(pb.createMultiBody(baseVisualShapeIndex=vis, basePosition=target, baseMass=0))

    def step(self, actions):
        for act, agent in zip(actions, self.agents):
            agent.step(act)

        self.environment_simulation(5)

        for agent in self.agents:
            agent.update_reward()
            agent.plan_update_request(self.boxes.boxes_id)

        rews = [agent.reward for agent in self.agents]
        self.stat.reward += sum(rews)

        if self.stat.step >= self.conf.environment_max_steps:
            self.done = [1, 1, 1]
        if len(self.boxes.boxes_id) == 0:
            self.done = [1, 1, 1]
            print('done')
        self.stat.step += 1

        if self.stat.change_plan:
            self.update_plan()

        state = self.get_state()

        # if self.stat.step % 5 == 0:
        #     self.plot_states(state)
        return [state, rews, self.done, self.stat]

    def update_pixel_maps(self):
        self.obst_map.fill(0)
        for agent in self.agents:
            agent.pos_map.fill(0)
            agent.tar_map.fill(0)

            (x, y, z), _ = pb.getBasePositionAndOrientation(agent.id)
            self.obst_map[int(y), int(x)] += 1
            agent.pos_map[int(y), int(x)] = 1

            tx, ty, tz = get_coord(agent.plan)
            agent.tar_map[int(ty), int(tx)] = 1

        self.boxes.update_box_map()

    def environment_simulation(self, steps):
        for i in range(steps):
            pb.stepSimulation()

    def get_state(self):
        self.update_pixel_maps()

        state = []
        for i, agent in enumerate(self.agents):
            fine = self.agents[0].get_fine_image()
            state.append(fine)
            state.append([agent.fork_hei()])
            state.append(np.stack((self.obst_map, agent.pos_map, agent.tar_map,), axis=0))

            # generate states for critic
            cri_fine_obs = fine
            cri_pix_obs = [self.obst_map, agent.pos_map, agent.tar_map]
            cri_den_obs = [[agent.fork_hei()]]
            for other_agent in self.agents:
                if other_agent.id != agent.id:
                    cri_pix_obs.append(other_agent.pos_map)
                    cri_pix_obs.append(other_agent.tar_map)
                    cri_den_obs.append([*other_agent.last_act.tolist()])
            cri_den_obs = cri_den_obs[0]  # + cri_den_obs[1]# + cri_den_obs[2]

            state.append(cri_fine_obs)
            state.append(cri_den_obs)
            state.append(np.stack(cri_pix_obs, axis=0))

        return state

    def reset(self):
        self.stat.reset()
        pb.resetSimulation()
        pb.setGravity(0, 0, -10, physicsClientId=self.client)
        pb.setPhysicsEngineParameter(fixedTimeStep=0.05)
        pb.setVRCameraState([9.55, 4.62, 4.68])

        self.create_world()
        self.shelves.reset()
        self.boxes.reset()
        self.reset_agents()

        self.boxes.event()
        self.plan.update_map_graph()
        self.update_plan()

        self.done = [0, 0, 0]
        # self.boxes.event()

        # pb.stepSimulation()

        return self.get_state()

    def ini_simulation(self):
        self.client = pb.connect(pb.GUI) if self.conf.nenv == 1 else pb.connect(pb.DIRECT)

    def create_world(self):
        x, y, z = self.conf.size[2], self.conf.size[1], 0.2
        coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2])
        visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2], rgbaColor=[0.81, 0.925, 0.845, 1])

        pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                           basePosition=[x / 2, y / 2, z / 2],
                           baseMass=0)

        x, y, z = self.conf.size[2], 0.01, 1
        coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2])
        visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2], rgbaColor=[0.81, 0.925, 0.845, 1])

        pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                           basePosition=[x / 2, 0, z / 2],
                           baseMass=0)

        x, y, z = self.conf.size[2], 0.01, 1
        coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2])
        visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2], rgbaColor=[0.81, 0.925, 0.845, 1])

        pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                           basePosition=[x / 2, self.conf.size[1], z / 2],
                           baseMass=0)

        x, y, z = 0.01, self.conf.size[1], 1
        coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2])
        visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2], rgbaColor=[0.81, 0.925, 0.845, 1])

        pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                           basePosition=[0, y / 2, z / 2],
                           baseMass=0)

        x, y, z = 0.01, self.conf.size[1], 1
        coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2])
        visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[x / 2, y / 2, z / 2], rgbaColor=[0.81, 0.925, 0.845, 1])

        pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                           basePosition=[self.conf.size[2], y / 2, z / 2],
                           baseMass=0)
