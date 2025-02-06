from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim

import omni.usd

from omni.isaac.core import World
import omni.kit

import numpy as np
import torch
import math



import omni


stage = omni.usd.get_context().get_stage()
             # Used to interact with simulation


from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.rotations import *


from omniisaacgymenvs.tasks import CubicSpline
import scipy.linalg as la
from numpy import genfromtxt
from omniisaacgymenvs.robots.articulations.hunter import Hunter
from omniisaacgymenvs.tasks.angle import angle_mod
from omniisaacgymenvs.tasks.LQRController import State

mydata = genfromtxt('/home/username/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/OmniIsaacGymEnvs/omniisaacgymenvs/Waypoints/Austin_centerline2.csv', delimiter = ',')
#mydata = genfromtxt('/home/username/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/OmniIsaacGymEnvs/omniisaacgymenvs/Waypoints/BrandsHatch_centerline.csv', delimiter = ',')
#mydata = genfromtxt('/home/username/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/omniisaacgymenvs/Waypoints/Silverstone_centerline.csv', delimiter = ',')
crosstrack_error = []
yaw_error = []
vel = []
xpos = []
ypos = []
LQR_Steer = []
Rl_Steer = []
pitch_local = []
xy_disturbed = []

class HunterTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._hunterse_positions = torch.tensor([0.0, 0.0, 0.4])
        self._hunterse_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0])

        
        self.max_episode_length = 1000

        self._num_observations = 7
        self._num_actions = 2
        
        
        RLTask.__init__(self, name, env)

        return
    
    def get_hunter(self):
        hunter = Hunter(prim_path=self.default_zero_env_path + "/Hunter", name="Hunter", translation=self._hunterse_positions,
                        orientation = self._hunterse_orientations)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Hunter", get_prim_at_path(hunter.prim_path), self._sim_config.parse_actor_config("Hunter"))

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Hunter USD file

        self._stage = get_current_stage()
        self.get_hunter()
        super().set_up_scene(scene)
        usd_path_terrain = "/home/username/Hybrid_Deep_Reinforcement_Learning_RoughTerrain/OmniIsaacGymEnvs/omniisaacgymenvs/USD_Files/Terrain_Files/terraintrain_9_uneven.usd"

        # Add rough terrain to the stage
        create_prim(prim_path="/World/Terrain", prim_type="Xform", position=[0.0, 0.0, 0.0])
        add_reference_to_stage(usd_path_terrain, "/World/Terrain")

        # create an ArticulationView wrapper for our Hunter - this can be extended towards accessing multiple Hunters
        self._hunterse = ArticulationView(prim_paths_expr="/World/envs/.*/Hunter/base_link", name="hunter_view", reset_xform_properties=False)
        # add Hunter ArticulationView and ground plane to the Scene
        self._base = RigidPrimView(prim_paths_expr="/World/envs/.*/Hunter/base_link",
            name="base_view5", reset_xform_properties=False, track_contact_forces=True)
        scene.add(self._hunterse)
        scene.add(self._base)
        #scene.add_default_ground_plane()

        self.ax = mydata[::10,0]
        self.ay = mydata[::10,1]
        self.cx, self.cy, self.cyaw, self.ck, self.s = CubicSpline.calc_spline_course(
         self.ax, self.ay, ds=0.1)
        
        # Uncomment the following for waypoints visualization

        # for i in range(len(mydata[:,0])):
        #     cube_idx = str(i)
        #     if i < len(mydata[:,0]) -1:
        #         qx = 0
        #         qy = 0
        #         yaw = math.atan2((mydata[i+1,1] - mydata[i,1]),

        #                         (mydata[i+1,0] - mydata[i,0])

        #         )
        #         qz = math.sin(yaw/2)
        #         w = math.cos(yaw/2)
        #     else:
        #         qx=0
        #         qy=0
        #         qz=0
        #         w=1        
        #     VisualCuboid(prim_path="/World/cube" + cube_idx,

        #         position=np.array([mydata[i , 0], mydata[i ,1], -0.6]),


        #         orientation=np.array([w,qx,qy,qz]),

        #         scale=np.array([0.5, 0.5, 0.25]),

        #         color=np.array([0,0,255]))

        self.Q = np.eye(4)         # LQR controller Q and R matrix
        self.Q[0,1] = 10.0
        self.Q[1,2] = 100.0
        self.Q[2,3] = 100.0
        self.R = np.eye(1)
        self.dt = 0.01
        self.L = 0.608
        self.max_iter = 150     # LQR controller DARE maximum iterations
        self.eps = 0.0167       # Epsilon value
        self.step_counter = 0.0
        self.delta_in = torch.zeros(self._num_envs, 1)             # Inside wheel steering angle
        self.delta_out = torch.zeros(self._num_envs, 1)            # Outside wheel steering angle

        self.crosstrack_error = torch.zeros((1, self._num_envs), dtype=torch.float32, device=self._device)
        self.yaw_error = torch.zeros((1, self._num_envs), dtype=torch.float32, device=self._device)
      
        self.default_dof_pos = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device, requires_grad=False)
        

    def get_observations(self):
        chassis_position, self.chassis_rotation = self._hunterse.get_world_poses(clone=False)  #Chassis position and orientation w.r.t world frame
        _, chassis_rotation_local = self._hunterse.get_local_poses()                           #Chassis position and orientation w.r.t body frame
        

        roll_angle, _, yaw = get_euler_xyz(chassis_rotation_local)
        _, _, yaw_world = get_euler_xyz(self.chassis_rotation)

        roll = torch.where(roll_angle >=torch.pi, roll_angle - 2*torch.pi, roll_angle)
        root_velocities = self._hunterse.get_velocities(clone=False)
        current_state = chassis_position[:, 0:2]
        velocity = root_velocities[:,0:3]



        base_lin_vel = torch.abs(quat_rotate_inverse(self.chassis_rotation, velocity)[:,0])    #Linear velocity w.r.t body frame


        x_pos = chassis_position[:,0]
        y_pos = chassis_position[:,1]


        # Error calculation by finding nearest waypoint w.r.t robot 
        for i in range(self._num_envs):
            self.alpha = yaw_world[i].cpu().numpy()
            
            state = State(x=x_pos[i].cpu().numpy(), y=y_pos[i].cpu().numpy(), yaw=yaw_world[i].cpu().numpy(), 
                          v=base_lin_vel[i].cpu().numpy())
        
            
            
            dx = np.subtract(state.x, self.cx)
            dy = np.subtract(state.y, self.cy)

            d = np.square(dx) + np.square(dy)
            mind = np.sqrt(np.min(d))
            
            self.ind = np.argmin(d)
            dxl = self.cx[self.ind] - state.x
            dyl = self.cy[self.ind] - state.y
            
            angle = angle_mod((self.cyaw[self.ind] - math.atan2(dyl, dxl)))
            if angle < 0:
                mind *= -1

            #mind = math.sqrt(mind)
            crosstrack_error = torch.from_numpy(np.asarray(mind))
           

            yaw_error = torch.from_numpy(np.asarray(angle_mod(state.yaw - self.cyaw[self.ind])))    
            
            self.obs_buf[i, 0] = x_pos[i]
            self.obs_buf[i, 1] = y_pos[i]
            self.obs_buf[i, 2] = roll[i] # Roll
            self.obs_buf[i, 3] = yaw_world[i]# Yaw
            self.obs_buf[i, 4] = base_lin_vel[i]
            self.obs_buf[i, 5] = crosstrack_error
            self.obs_buf[i, 6] = yaw_error
       
        observations = {
            self._hunterse.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations  
      
    def post_reset(self):

        self._rwl_dof_idx = self._hunterse.get_dof_index("re_left_jiont")   #DOF index of rear wheel left                           
        self._rwr_dof_idx = self._hunterse.get_dof_index("re_right_jiont")  #DOF index of rear wheel right
        self._fsr_dof_idx = self._hunterse.get_dof_index("fr_steer_left_joint")   #DOF index of front steer right joint                           
        self._fsl_dof_idx = self._hunterse.get_dof_index("fr_steer_right_joint")  #DOF index of front steer left joint
        self._front_steer_joint_idx = self._hunterse.get_dof_index("front_steer_joint")
        
        self.root_velocities = self._hunterse.get_velocities(clone=False)
        self.dof_pos = self._hunterse.get_joint_positions(clone=False)
        self.dof_vel = self._hunterse.get_joint_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = self._hunterse.get_world_poses()
        

        # randomize all envs
        indices = torch.arange(self._hunterse.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def disturb_robot(self):
        # Adding random disturbances as linear and angular velocties w.r.t world frame

        self.root_velocities[:, 0:2] = torch.randint(
            -3,3,(self._num_envs,2), device=self._device
        )
        self.root_velocities[:,5] = torch.randint(-4,4,(self._num_envs,1), device=self._device
                                                     )
        self._hunterse.set_velocities(self.root_velocities)

    def reset_idx(self, env_ids=None):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)
        velocities = torch.zeros((num_resets, self._hunterse.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities
        dof_pos[:, self._rwl_dof_idx] = torch.zeros((num_resets), device=self._device) 
        dof_pos[:, self._rwr_dof_idx] = torch.zeros((num_resets), device=self._device)
        dof_vel[:, self._fsl_dof_idx] = torch.zeros((num_resets), device=self._device)
        dof_vel[:, self._fsr_dof_idx] = torch.zeros((num_resets), device=self._device)

        

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        self._hunterse.set_joint_positions(dof_pos, indices)
        
        
        self._hunterse.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices)
        self._hunterse.set_velocities(root_vel, indices)

        #Bookeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def new_dof_state_tensors(self):
        self.dof_pos = self._hunterse.get_joint_positions(clone=False)
        self.dof_vel = self._hunterse.get_joint_velocities(clone=False)
    
    def new_body_state_tensors(self):
        self.base_pos, self.base_quat = self._hunterse.get_world_poses(clone=False)
        self.base_velocities = self._hunterse.get_velocities(clone=False)    
        
    def pre_physics_step(self, actions) -> None:

        
        ##

        if not self._env._world.is_playing():
            return
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        indices = torch.arange(self._hunterse.count, dtype=torch.int32, device=self._device)
        actions = actions.to(self._device)
        self.position = torch.zeros((self._hunterse.count, self._hunterse.num_dof), dtype=torch.float32, device=self._device)
        self.vels = torch.zeros((self._hunterse.count, self._hunterse.num_dof), dtype=torch.float32, device=self._device)
        
        
        for i in range(self._num_envs):
            state = State(x=self.obs_buf[i, 0].cpu().numpy(), y=self.obs_buf[i, 1].cpu().numpy(), yaw=self.obs_buf[i,3].cpu().numpy(), 
                          v=self.obs_buf[i,4].cpu().numpy())
                

            # Solve Discrete Time Riccati Equation      
                
            x = self.Q
            x_next = self.Q
            
            
            v = state.v
            
            dx = np.subtract(state.x, self.cx)
            dy = np.subtract(state.y, self.cy)

            d = np.square(dx) + np.square(dy)
            mind = np.min(d)
            self.ind = np.argmin(d)

            A = np.zeros((4, 4))
            A[0, 0] = 1.0
            A[0, 1] = self.dt
            A[1, 2] = v
            A[2, 2] = 1.0
            A[2, 3] = self.dt
            # print(A)

            B = np.zeros((4, 1))
            B[3, 0] = v / self.L

            for j in range(self.max_iter):
                x_next = A.T @ x @ A - A.T @ x @ B @ \
                        la.inv(self.R + B.T @ x @ B) @ B.T @ x @ A + self.Q
                if (abs(x_next - x)).max() < self.eps:
                    break
                x = x_next

            X = x_next
            K = la.inv(B.T @ X @ B + self.R) @ (B.T @ X @ A)

            # Formalize LQR controller
            k = self.ck[self.ind]
            v = state.v
            th_e = self.obs_buf[i,6].cpu().numpy()

            x = np.zeros((4, 1))
            e = self.obs_buf[i, 5].cpu().numpy()
            pe = 0.0
            pth_e = 0.0
            x[0, 0] = e
            x[1, 0] = (e - pe) / self.dt
            x[2, 0] = th_e
            x[3, 0] = (th_e - pth_e) / self.dt

            ff = math.atan2(self.L * k, 1)
            fb = angle_mod((-K @ x)[0, 0])

            steer = ff + fb
            LQR_Steer.append(np.asarray(steer))
            np.savetxt("LQR_Steer.csv" , LQR_Steer, delimiter=',')


            self.position[i, self._front_steer_joint_idx] = torch.clamp(steer + actions[i, 0], min=-0.524, max=0.524)  #Combining outputs of both controllers
                   
        Rl_Steer.append(actions[0,0].cpu().numpy())
        np.savetxt("RL_Steer.csv" , Rl_Steer, delimiter=',')

        actions[:, 1] = torch.clamp(21.82*actions[:, 1], min=5.0, max=21.82)
        
        self.vels[:, self._rwl_dof_idx] = actions[:,1]                                           
        self.vels[:, self._rwr_dof_idx] = actions[:,1]   
        
        
        
        # Apply ackermann geometry
        
        self.delta_out = torch.atan(0.608*torch.tan(self.position[:,self._front_steer_joint_idx])/
                                    (0.608 + 0.5*0.554*torch.tan(self.position[:,self._front_steer_joint_idx])))
        
        self.delta_in = torch.atan(0.608*torch.tan(self.position[:,self._front_steer_joint_idx])/
                                    (0.608 - 0.5*0.554*torch.tan(self.position[:,self._front_steer_joint_idx])))
        
        self.position[:, self._fsr_dof_idx] = torch.where(self.position[:, self._front_steer_joint_idx] <= 0, self.delta_in, self.delta_out)
        
        self.position[:, self._fsl_dof_idx] = torch.where(self.position[:, self._front_steer_joint_idx] > 0, self.delta_in, self.delta_out) 


        
        for i in range(10): # Output frequency = dt*num_iterations, 10Hz for Hunter
            self._hunterse.set_joint_velocity_targets(self.vels, indices=indices)
            self._hunterse.set_joint_position_targets(self.position, indices=indices)
            SimulationContext.step(self.world, render=True)
            self.new_dof_state_tensors()
            self.new_body_state_tensors()

    def calculate_metrics(self) -> None:
        self.step_counter += 1.0
        
        chassis_position, chassis_rotation = self._hunterse.get_world_poses(clone=False)
        _, chassis_rotation_local = self._hunterse.get_local_poses()
    
        self.x_pos = chassis_position[:,0]
        self.y_pos = chassis_position[:,1]

        
        roll_angle, pitch_c, _ = get_euler_xyz(chassis_rotation_local)
        pitch_c = torch.where(pitch_c>=torch.pi, pitch_c - 2*torch.pi, pitch_c)

        _, _, yaw_world = get_euler_xyz(chassis_rotation)


        roll = torch.where(roll_angle >=torch.pi, roll_angle - 2*torch.pi, roll_angle)
        root_velocities = self._hunterse.get_velocities(clone=False)

        root_velocities = self._hunterse.get_velocities(clone=False)
        velocity = root_velocities[:,0:3]
    

        self.base_lin_vel = torch.abs(quat_rotate_inverse(chassis_rotation, velocity)[:,0])
        self.has_fallen5 = (torch.norm(self._base.get_net_contact_forces(clone=False), dim=1) > 1.)
        


        x_pos = chassis_position[:,0]
        y_pos = chassis_position[:,1]

       

        for i in range(self._num_envs):
            self.alpha = yaw_world[i].cpu().numpy()
            
            state = State(x=x_pos[i].cpu().numpy(), y=y_pos[i].cpu().numpy(), yaw=yaw_world[i].cpu().numpy(), 
                          v=self.base_lin_vel[i].cpu().numpy())
        
            
            
            dx = np.subtract(state.x, self.cx)
            dy = np.subtract(state.y, self.cy)

            d = np.square(dx) + np.square(dy)
            mind = np.sqrt(np.min(d))
            
            self.ind = np.argmin(d)
            dxl = self.cx[self.ind] - state.x
            dyl = self.cy[self.ind] - state.y
            
            angle = angle_mod((self.cyaw[self.ind] - math.atan2(dyl, dxl)))
            if angle < 0:
                mind *= -1

            #mind = math.sqrt(mind)
            self.crosstrack_error[0,i] = torch.from_numpy(np.asarray(mind))

            self.yaw_error[0, i] = torch.from_numpy(np.asarray(angle_mod(state.yaw - self.cyaw[self.ind])))
        
        x_pos_numpy = x_pos.cpu().numpy()
        y_pos_numpy = y_pos.cpu().numpy()

        ##

        # Uncomment following to add disturbances during training/evaluation phase

        # if self.step_counter%100==0:
        #     self.disturb_robot()
        #     for x,y in zip(x_pos_numpy, y_pos_numpy):
        #         xy_disturbed.append((x,y))  
        #     print("Disturbed")

        ##

        vel_numpy = self.base_lin_vel.cpu().numpy()
        crosstrack_numpy = self.crosstrack_error.cpu().numpy()[0,0]
        yaw_numpy = self.yaw_error.cpu().numpy()[0,0]
        pitch_numpy = pitch_c.cpu().numpy()

        crosstrack_error.append(crosstrack_numpy)
        yaw_error.append(yaw_numpy)
        xpos.append(x_pos_numpy)
        ypos.append(y_pos_numpy)
        vel.append(vel_numpy)
        pitch_local.append(pitch_numpy)

        np.savetxt("X_pos_numpy.csv" , xpos, delimiter=',')
        np.savetxt("Y_pos_numpy.csv" , ypos, delimiter=',')
        np.savetxt("vel_numpy.csv" , vel, delimiter=',')
        np.savetxt("Crosstrack_numpy.csv" , crosstrack_error, delimiter=',')
        np.savetxt("Yaw_numpy.csv" , yaw_error, delimiter=',')
        np.savetxt("Yaw_numpy.csv" , yaw_error, delimiter=',')
        np.savetxt("Pitch_local.csv" , pitch_local, delimiter=',')
        np.savetxt("XY Disturbed", xy_disturbed, delimiter=',')    

        self.crosstrack_error -= 0.0
        self.crosstrack_error /= 10.0

        self.yaw_error -= 0
        self.yaw_error /= 3.14

        self.base_lin_vel -= 0.0
        self.base_lin_vel /= 3.0
        reward_speed = self.base_lin_vel

        #Calculating exponential rewards
        reward_crosstrack_error = torch.where(self.crosstrack_error >= 0, torch.exp(-1.0*self.crosstrack_error), torch.exp(self.crosstrack_error))
        reward_yaw_error = torch.where(self.yaw_error >=0 , torch.exp(-1.0*self.yaw_error), torch.exp(1.0*self.yaw_error))
        reward_reset = torch.where((self.progress_buf <= self.max_episode_length) & (self.reset_buf == 1.0), -1.0, 0.0)

        
        

        reward = reward_crosstrack_error*reward_yaw_error*reward_speed + reward_reset
        
        
       

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        
        self.step_counter = torch.from_numpy(np.asarray(self.step_counter))
        
        resets = torch.where(torch.abs(self.obs_buf[:,5]) >= 10, 1, 0)
        resets = torch.where((self.base_lin_vel[:] <= 0.01) & (self.step_counter%50==0), 1, resets)
        resets = torch.where(self.has_fallen5==True, 1, resets)  #Reset if the robot flips over

        self.reset_buf[:] = resets          
