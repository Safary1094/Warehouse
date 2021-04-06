import networkx as nx
# from _collections import deque
from Config import Config
from Entities import Agent, Boxes, Shelves, get_coord, get_ang
import matplotlib.pyplot as plt
import random
import heapq
import copy
import pybullet as pb
import numpy as np

plt.ion()


class PlanGraph:
    def __init__(self):
        self.agents = set()
        self.free_agents = set()

        self.free_dest = []

        self.edges = []
        self.acts = {}
        # self.time = 0
        self.dist = 0
        self.neg_reward = 0
        self.finish = False


class Plan:
    def __init__(self, agent, boxes: Boxes, shelves: Shelves, conf: Config):
        self.agents = agent
        self.boxes = boxes
        self.shelves = shelves
        self.conf = conf
        self.plans_list = []

    def expand_plan(self):
        plan_total_dist = 0
        finish = False
        while not (finish or plan_total_dist > 100):
            current_plan = heapq.heappop(self.plans_list)[-1]
            plan_expand_list = self.plan_step(current_plan)
            a = 1
            for plan in plan_expand_list:
                plan_total_dist = plan.dist
                plan_total_reward = plan.neg_reward
                finish = max(finish, plan.finish)

                if len(self.plans_list) > self.conf.mc_num:
                    heapq.heappushpop(self.plans_list, (plan_total_dist + plan_total_reward, random.random(), plan))
                else:
                    heapq.heappush(self.plans_list, (plan_total_dist + plan_total_reward, random.random(), plan))
                a = 1

        return

    def update_map_graph(self):
        map = nx.Graph()
        nodes = []
        for i in range(self.conf.size[1]):
            for j in range(self.conf.size[2]):
                nodes.append((i, j))
        map.add_nodes_from(nodes)

        direct_edges = []
        diag_edges = []
        no_edges = []
        # add static_map
        stat_map = np.zeros(self.conf.size)
        for (i, j) in nodes:
            if stat_map[0, i, j]:
                continue

            if i > 0:
                if stat_map[0, i - 1, j]:
                    no_edges.append(((i, j), (i - 1, j)))
                else:
                    direct_edges.append(((i, j), (i - 1, j)))

            if i < self.conf.size[0] - 1:
                if stat_map[0, i + 1, j]:
                    no_edges.append(((i, j), (i + 1, j)))
                else:
                    direct_edges.append(((i, j), (i + 1, j)))

            if j > 0:
                if stat_map[0, i, j - 1]:
                    no_edges.append(((i, j), (i, j - 1)))
                else:
                    direct_edges.append(((i, j), (i, j - 1)))

            if j < self.conf.size[1] - 1:
                if stat_map[0, i, j + 1]:
                    no_edges.append(((i, j), (i, j + 1)))
                else:
                    direct_edges.append(((i, j), (i, j + 1)))

            if i > 0 and j > 0:
                if stat_map[0, i - 1, j - 1]:
                    no_edges.append(((i, j), (i - 1, j - 1)))
                else:
                    diag_edges.append(((i, j), (i - 1, j - 1)))
            if i < self.conf.size[1] - 1 and j < self.conf.size[2] - 1:
                if stat_map[0, i + 1, j + 1]:
                    no_edges.append(((i, j), (i + 1, j + 1)))
                else:
                    diag_edges.append(((i, j), (i + 1, j + 1)))
            if i > 0 and j < self.conf.size[2] - 1:
                if stat_map[0, i - 1, j + 1]:
                    no_edges.append(((i, j), (i - 1, j + 1)))
                else:
                    diag_edges.append(((i, j), (i - 1, j + 1)))
            if i < self.conf.size[1] - 1 and j > 0:
                if stat_map[0, i + 1, j - 1]:
                    no_edges.append(((i, j), (i + 1, j - 1)))
                else:
                    diag_edges.append(((i, j), (i + 1, j - 1)))

        map.add_edges_from(direct_edges, weight=1)
        map.add_edges_from(diag_edges, weight=1.4142)
        map.add_edges_from(no_edges, weight=1000)
        a = 1

        self.dist_mat = nx.floyd_warshall(map)

    def get_distance(self, a, b, mobile=False):
        begin_coord = (int(a[1]), int(a[0]))
        end_coord = (int(b[1]), int(b[0]))
        dist = self.dist_mat[begin_coord][end_coord]
        return dist

    def initial_graph(self):

        plan_graph = PlanGraph()
        plan_graph.agents = set([(agent.id, get_coord(agent.id)) for agent in self.agents])
        plan_graph.free_agents = set([(agent.id, get_coord(agent.id)) for agent in self.agents])

        plan_graph.dest = set([(box_id, None) for box_id in self.boxes.boxes_id])
        plan_graph.free_dest = [(box_id, None) for box_id in self.boxes.boxes_id]

        plan_graph.acts = {agent.id: [] for agent in self.agents}

        self.plans_list.clear()
        heapq.heappush(self.plans_list, (0, random.random(), plan_graph))

        a = 1

    def draw(self, graph):
        # self.draw(self.plans_list[0].graph)
        nx.draw(graph)
        plt.show()
        plt.pause(1)

    def plan_step(self, old_plan: PlanGraph):
        plan_expand_list = []
        # remove shortest edge
        if len(old_plan.free_agents) == 0:
            min_dist, min_agent, min_dest = min(old_plan.edges)
            old_plan.free_agents.add((min_agent[0], min_dest[1]))
            old_plan.edges.remove((min_dist, min_agent, min_dest))
            old_plan.edges = [(edge[0] - min_dist, edge[1], edge[2]) for edge in old_plan.edges]

        node_start = old_plan.free_agents.pop()

        for node_end in old_plan.free_dest + [node_start]:
            plan = copy.deepcopy(old_plan)
            if node_end != node_start:
                plan.free_dest.remove(node_end)
            if node_end[1] is None:
                coord, _ = pb.getBasePositionAndOrientation(node_end[0])
                node_end = (node_end[0], coord)

            new_edge_dist = self.get_distance(node_start[1], node_end[1])

            if node_end is not node_start:
                plan.neg_reward -= self.conf.rewards['reach_box']

            plan.edges.append((new_edge_dist, node_start, node_end))
            plan.acts[node_start[0]].append(node_end)
            plan.dist += new_edge_dist
            if len(plan.free_dest) == 0 and len(plan.free_agents) == 0:
                plan.finish = True
            plan_expand_list.append(plan)

        return plan_expand_list

    def find_best_plan(self):
        return self.plans_list[-1]
