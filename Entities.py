import numpy as np
from random import randint as ri
import random
from Config import Config
import pybullet as pb

import matplotlib.pyplot as plt

plt.ion()


def get_coord(body_id, link_id=None):
    if link_id is None:
        coord, _ = pb.getBasePositionAndOrientation(body_id)
    else:
        coord = pb.getLinkState(body_id, link_id)[0]
    return coord


def get_ang(body_id, link_id=None):
    if link_id is None:
        pose, ang = pb.getBasePositionAndOrientation(body_id)
    else:
        i = pb.getLinkState(body_id, link_id)
        ang = i[1]
    ang = pb.getMatrixFromQuaternion(ang)
    return ang[0], ang[3]


def get_velo(id):
    velo = pb.getBaseVelocity(id)
    return velo[0][0], velo[0][1]


class Stat:
    def __init__(self):
        self.collision = 0
        self.motions = np.array([0.0, 0.0, 0.0])
        self.change_plan = True
        self.step = 0
        self.reward = 0.0
        self.boxes = 0
        self.step_to_target = 0

    def reset(self):
        self.collision = 0
        self.motions.fill(0)
        self.change_plan = True
        self.step = 0
        self.reward = 0.0
        self.boxes = 0
        self.step_to_target = 0


class Agent:
    def __init__(self, urdf, conf: Config, stat: Stat):
        self.urdf = urdf
        self.conf = conf

        self.id = None
        self.plan = None

        self.prev_target_dist = None
        self.prev_box_dir = None
        self.plan_mark_id = []
        self.carry_box = False
        self.pos_map = np.zeros((self.conf.size[1], self.conf.size[2]))
        self.tar_map = np.zeros((self.conf.size[1], self.conf.size[2]))
        self.stat = stat
        self.reward = 0.0
        self.fork_id = 2

    def get_fine_image(self):
        # return np.zeros((1, self.conf.camera_res, self.conf.camera_res))
        pos, ang = pb.getBasePositionAndOrientation(self.id)
        camera_pos = [pos[0], pos[1], pos[2] + 2]
        target_pos = [pos[0], pos[1], 0]
        ang = pb.getMatrixFromQuaternion(ang)
        cameraUpVector = [ang[0], ang[3], ang[6]]
        view_matrix = pb.computeViewMatrix(cameraEyePosition=camera_pos,
                                           cameraTargetPosition=target_pos,
                                           cameraUpVector=cameraUpVector)

        projection_matrix = pb.computeProjectionMatrixFOV(fov=100.0, aspect=1.0, nearVal=0.1, farVal=5.1)
        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(width=self.conf.camera_res,
                                                                    height=self.conf.camera_res,
                                                                    viewMatrix=view_matrix,
                                                                    projectionMatrix=projection_matrix)

        gray = 0.2989 * rgbImg[:, :, 0] + 0.5870 * rgbImg[:, :, 1] + 0.1140 * rgbImg[:, :, 2]
        gray = np.expand_dims(gray, axis=0)
        return gray

    def collision(self):
        if self.id is None:
            return True

        for c in pb.getContactPoints(self.id):
            if c[2] != 0:  # if collision not with ground
                return True

        return False

    def reset(self):
        self.carry_box = False
        if self.id is not None:
            pb.removeBody(self.id)

        dz = 0.8
        dy = random.random() * self.conf.size[2]
        dx = random.random() * self.conf.size[1]
        orientation = pb.getQuaternionFromEuler([0, 0, random.random() * 2 * np.pi])
        self.id = pb.loadURDF(self.urdf, basePosition=[dx, dy, dz], baseOrientation=orientation)
        self.plan_mark_id = []
        self.plan = self.id
        self.last_act = np.zeros(3)

    def fork_hei(self):
        fork_id = 2
        return pb.getLinkState(self.id, fork_id)[0][2]

    def fork_2_plan_distance(self):
        fork_pos = np.array(get_coord(self.id, self.fork_id))
        d = np.sqrt(np.sum((np.array(get_coord(self.plan)) - fork_pos) ** 2))
        return d

    def fork_2_box_ang(self):
        fork_ang = np.array(get_ang(self.id, self.fork_id))
        tar_ang = np.array(get_ang(self.plan))

        fork_pos = np.array(get_coord(self.id, self.fork_id))
        tar_pos = np.array(get_coord(self.plan))

        d1 = abs(np.dot(fork_ang, tar_ang))  # 0 - bad, 1 - good

        tar_dir = ((tar_pos - fork_pos)[0:2])
        tar_dir /= np.linalg.norm(tar_dir)
        d2 = np.dot(fork_ang, tar_dir)  # 0 - bad, 2 - good

        return d1 * d2

    def step(self, act):
        self.last_act = act
        self.reward = 0 * abs(act).sum()
        pb.setJointMotorControlArray(self.id, [0, 1, 2], pb.VELOCITY_CONTROL, targetVelocities=act,
                                     forces=[500, 500, 10])
        self.stat.motions += act

    def plan_update_request(self, boxes_id):
        if self.plan is None:
            self.stat.change_plan += True
            return

        if self.plan != self.id and self.fork_2_plan_distance() < 0.2:
            pb.removeBody(self.plan)
            boxes_id.remove(self.plan)
            self.stat.change_plan += True
            self.prev_target_dist = None

    def update_reward(self):
        # step to/from plan
        target_dist = self.fork_2_plan_distance()
        if self.prev_target_dist is None:
            self.prev_target_dist = target_dist

        if target_dist > 1.3:
            self.reward += 50 * (self.prev_target_dist - target_dist)
        else:
            self.reward += 25 * (self.prev_target_dist - target_dist)
        self.prev_target_dist = target_dist

        # angular penalty
        box_dir = self.fork_2_box_ang()
        if self.prev_box_dir is None:
            self.prev_box_dir = box_dir
        if target_dist < 1.3:
            self.reward += 25 * (box_dir - self.prev_box_dir)
        self.prev_box_dir = box_dir

        # collision penalty
        if self.collision():
            if self.fork_2_plan_distance() < 1 and box_dir > 0.9:
                pass
            else:
                self.reward += self.conf.rewards['collision_penalty']
                self.stat.collision += 1

        # reach plan reward
        if target_dist < 0.2 and self.plan != self.id:
            self.reward += 15  # * self.conf.rewards['reach_box']
            self.stat.boxes += 1


class Boxes:
    def __init__(self, conf: Config, obst_map):
        self.conf = conf
        self.box_map = obst_map
        self.zone_pixel_map = np.zeros((1, self.conf.size[1], self.conf.size[2]))

        self.boxes_id = set()
        self.out_box = set()
        self.stored_box = set()

        self.in_zone = []
        self.out_zone = []

    def draw_zone_pixel_map(self):
        pass

    def update_box_map(self):
        self.box_map.fill(0)
        for box_id in self.boxes_id:
            (x, y, z), _ = pb.getBasePositionAndOrientation(box_id)
            self.box_map[int(y), int(x)] += 1

    def event(self):
        if len(self.boxes_id) == 0:
            self.event_in(self.conf.box_amount)
        # if len(self.out_box) == 0:
        #     self.event_out()

    def event_in(self, num):
        placed = 0
        tries = 0
        while not (placed >= num or tries > 30):
            dz = 0.3
            dy = random.random() * self.conf.size[2]
            dx = random.random() * self.conf.size[1]
            orientation = pb.getQuaternionFromEuler([0, 0, random.random() * 2 * np.pi])

            box_id = pb.loadURDF(self.conf.box_path, basePosition=[dx, dy, dz], baseOrientation=orientation)
            pb.stepSimulation()

            if pb.getContactPoints(box_id):
                pb.removeBody(box_id)
            else:
                self.boxes_id.add(box_id)
                placed += 1

    def reset(self):

        self.boxes_id.clear()
        self.out_box.clear()
        self.stored_box.clear()


class Shelves:
    def __init__(self, conf: Config):
        self.conf = conf
        self.size = self.conf.size
        self.map = np.zeros(self.conf.size)

        self.vacant_pos = []
        self.shelves = []

    def reset(self):
        self.shelves.clear()
        self.vacant_pos.clear()
        self.map = np.zeros(self.conf.size)
        # self.map.fill(0)
        # generate params:
        while True:
            dir = random.choice(['ver'])
            sh1_length = random.choice([7, 6, 5, 4])
            sh2_length = random.choice([7, 6, 5, 4])
            sh1_hei = random.choice(self.conf.shelves_height)
            sh2_hei = random.choice(self.conf.shelves_height)
            row_num = random.choice(self.conf.shelves_row_num)
            y_clearence = 3
            x_clearance = 4

            y_start = ri(0, 3)
            x_start = ri(5, 8)
            if (y_start + y_clearence + sh1_length + sh2_length) < self.conf.size[1]:
                break

        if dir == 'ver':
            for r_i in range(row_num):
                y0 = y_start
                x0 = x_start + r_i * x_clearance
                if self.add_shelf_to_sim(x0, x0, y0, y0 + sh1_length, sh1_hei, dir):
                    self.add_shelf_to_list(x0, x0, y0, y0 + sh1_length, sh1_hei, dir)

                x0 += 1
                if self.add_shelf_to_sim(x0, x0, y0, y0 + sh1_length, sh1_hei, dir):
                    self.add_shelf_to_list(x0, x0, y0, y0 + sh1_length, sh1_hei, dir)

                y0 += sh1_length + y_clearence
                x0 = x_start + r_i * x_clearance
                if self.add_shelf_to_sim(x0, x0, y0, y0 + sh2_length, sh2_hei, dir):
                    self.add_shelf_to_list(x0, x0, y0, y0 + sh2_length, sh2_hei, dir)

                x0 += 1
                if self.add_shelf_to_sim(x0, x0, y0, y0 + sh2_length, sh2_hei, dir):
                    self.add_shelf_to_list(x0, x0, y0, y0 + sh2_length, sh2_hei, dir)

            if dir == 'hor':
                pass

        self.add_shelves_to_pixel()

    def add_shelf_to_sim(self, x0, x1, y0, y1, sh_hei, dir):
        if dir == 'ver':
            xc = x0 + 0.5
            yc = 0.5 * (y0 + y1) + 0.5
            for zc in range(sh_hei):
                coll = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.9 * 0.5, (y1 - y0) * 0.5, 0.1 * 0.5])
                visu = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.9 * 0.5, (y1 - y0) * 0.5, 0.1 * 0.5])

                id = pb.createMultiBody(baseCollisionShapeIndex=coll, baseVisualShapeIndex=visu,
                                        basePosition=[xc, yc, zc + 0.1],
                                        baseMass=0)

                pb.stepSimulation()
                if pb.getContactPoints(id):
                    pb.removeBody(id)
                    return False

        return True

    def add_shelf_to_list(self, x0, x1, y0, y1, sh_hei, dir):
        if dir == 'ver':
            self.shelves.append([x0, x1, y0, y1, sh_hei, dir])
            for h in range(sh_hei):
                for y in range(y0, y1 + 1):
                    self.vacant_pos.append((h, y, x0))

    def add_shelves_to_pixel(self):
        v = np.array(self.vacant_pos)
        # self.map[v[:, 0], v[:, 1], v[:, 2]] = 1
        self.map = np.expand_dims(self.map[0], 0)
