import json
import numpy as np


class Config:
    def __init__(self, work_folder):
        file_name = open(work_folder + '/config.json')
        self.work_folder = work_folder
        self.config = json.load(file_name)

        self.size = self.get_size()
        self.lift_n = self.config['lifters_number']
        self.move_n = self.config['movers_number']
        self.architecture = self.get_architecture()

        self.nenv = self.config['learning_parameters']['envs_number']
        self.train_epochs = self.config['learning_parameters']['train_epochs']
        self.lr = self.config['learning_parameters']['lr']
        self.traj_len = self.config['learning_parameters']['trajectory_length']
        self.batch_size = self.config['learning_parameters']['batch_size']
        self.opt_epochs = self.config['learning_parameters']['opt_epochs']
        self.gam = self.config['learning_parameters']['gamma']
        self.lam = self.config['learning_parameters']['lambda']
        self.ent_coef = self.config['learning_parameters']['ent_coef']
        self.environment_max_steps = self.config['learning_parameters']['environment_max_steps']
        self.inference = self.config['learning_parameters']['inference']
        self.load_model = self.config['learning_parameters']['load_model']


        self.act_space = self.config['act_space']
        self.camera_res = self.config['camera_res']
        self.shelves_height = self.config['shelves_height']
        self.shelves_row_num = self.config['shelves_row_num']
        self.box_amount = self.config['box_amount']
        self.mc_num = self.config['mc_num']

        self.move_blue_path = self.config['move_blue_path']
        self.move_yellow_path = self.config['move_yellow_path']
        self.lift_red_path = self.config['lift_red_path']
        self.box_path = self.config['box_urdf']
        self.save_every_steps = self.config['save_every_steps']

        self.rewards = self.config['rewards']

    def get_size(self):
        return np.array(self.config['size'])

    def get_architecture(self):
        return self.config['architecture']
