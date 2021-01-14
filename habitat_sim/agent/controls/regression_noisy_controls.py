#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pickle import load

import attr
import habitat_sim.bindings as hsim
import magnum as mn
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import habitat_sim
from habitat_sim.utils.common import quat_rotate_vector, quat_from_two_vectors, cartesian_to_polar, agent_state_target2ref, quat_from_magnum, quat_from_angle_axis, quaternion_xyzw_to_wxyz
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.agent.controls.regression_models.actuation_nn import regress_3layer
from habitat_sim.registry import registry
from scipy.spatial.transform import Rotation as R
from sklearn import preprocessing
from torch.autograd import Variable
from collections import OrderedDict
import quaternion
from torch.nn.parallel import DistributedDataParallel as DDP
import os

@attr.s(auto_attribs=True)
class _ActuationModel:
    model_pth: str
    def __attrs_post_init__(self):
        #base_pth = '/private/home/jtruong/repos/habitat-sim/habitat_sim/agent/controls/'
        base_pth = '/srv/share3/jtruong33/develop/habitat-sim/habitat_sim/agent/controls/'
        self.x_scaler = load(open(base_pth + 'scalers/x_scaler_' + self.model_pth.split('.pt')[0] + '.pkl', 'rb'))
        self.y_scaler = load(open(base_pth + 'scalers/y_scaler_' + self.model_pth.split('.pt')[0] + '.pkl', 'rb'))
        if os.environ.get("SLURM_JOBID", None) is not None:
            local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            local_rank = 0
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.regress_noise_model = regress_3layer(7,4)
        self.regress_noise_model.to(self.device)
        checkpoint = torch.load(base_pth + 'regression_models/' + self.model_pth, map_location='cpu')
        self.regress_noise_model.load_state_dict(checkpoint)
        self.regress_noise_model.eval()

    def sample(self, position, rotation, action):

        heading_vector = quat_rotate_vector(rotation.inverse(), np.array([0, 0, -1]))
        ref_t = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]

        rotation_mp3d_habitat = quat_from_two_vectors(habitat_sim.geo.GRAVITY, np.array([0, 0, -1]))
        pt_mp3d = quat_rotate_vector(rotation_mp3d_habitat, position)

        input_xyt = [pt_mp3d[0], pt_mp3d[1], ref_t]
        input_xycs = [pt_mp3d[0], pt_mp3d[1], np.cos(ref_t), np.sin(ref_t)]

        processed_input_xycs = input_xycs.copy()
        processed_input_xycs[0] = np.array(self.x_scaler.transform(input_xycs[0].reshape(1, -1))).flatten()[0]
        processed_input_xycs[1] = np.array(self.y_scaler.transform(input_xycs[1].reshape(1, -1))).flatten()[0]
        processed_input_xycs.extend(action)

        in_tensor = torch.from_numpy(np.array(processed_input_xycs)).float()
        with torch.no_grad():
            data = Variable(in_tensor)
            if self.cuda:
                self.regress_noise_model.cuda()
                data = data.cuda()
            output = self.regress_noise_model(data).cpu().detach().numpy().tolist()
        output_xyt = [output[0], output[1], np.arctan2(output[3], output[2])]
        # position is in x, y, t
        ref_pos = [position, rotation]

        target_pos_xyt = np.array(input_xyt) + np.array(output_xyt)
        target_pos_xyz = [target_pos_xyt[0], target_pos_xyt[1], pt_mp3d[2]]

        pos_rotation_mp3d_habitat = quat_from_two_vectors(habitat_sim.geo.GRAVITY, np.array([0, 0, 1]))
        target_pt_mp3d = quat_rotate_vector(pos_rotation_mp3d_habitat, target_pos_xyz)

        target_pos_t = quat_from_angle_axis(output_xyt[-1], habitat_sim.geo.UP)
        target_pos = [target_pt_mp3d, target_pos_t]
        [position_in_ref_coordinate, rotation_in_ref_coordinate] = agent_state_target2ref(ref_pos, target_pos)
        noisy_trans = quat_rotate_vector(rotation_mp3d_habitat, position_in_ref_coordinate)
        target_quat = quaternion.as_float_array(quaternion_xyzw_to_wxyz(rotation_in_ref_coordinate))
        rot_noise = output_xyt[-1]
        return [noisy_trans[0], noisy_trans[1], rot_noise] # x y t

@attr.s(auto_attribs=True)
class MotionNoiseModel:
    motion: _ActuationModel


@attr.s(auto_attribs=True)
class ControllerNoiseModel:
    motion_noise: MotionNoiseModel

@attr.s(auto_attribs=True)
class RobotNoiseModel:
    ILQR_5k: ControllerNoiseModel
    ILQR_1k: ControllerNoiseModel
    ILQR_500: ControllerNoiseModel
    ILQR_250: ControllerNoiseModel
    ILQR_100: ControllerNoiseModel
    Proportional_5k: ControllerNoiseModel
    Proportional_1k: ControllerNoiseModel
    Proportional_500: ControllerNoiseModel
    Proportional_250: ControllerNoiseModel
    Proportional_100: ControllerNoiseModel
    Movebase_5k: ControllerNoiseModel
    Movebase_1k: ControllerNoiseModel
    Movebase_500: ControllerNoiseModel
    Movebase_250: ControllerNoiseModel
    Movebase_100: ControllerNoiseModel

    def __getitem__(self, key):
        return getattr(self, key)


r"""
Parameters contributed from PyRobot
https://pyrobot.org/
https://github.com/facebookresearch/pyrobot

Please cite PyRobot if you use this noise model
"""
noise_models = {
    "LoCoBot": RobotNoiseModel(
        ILQR_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_5k_3.pt')
            ),
        ),
        ILQR_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_1k_3.pt')
            ),
        ),
        ILQR_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_500_3.pt')
            ),
        ),
        ILQR_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_250_3.pt')
            ),
        ),
        ILQR_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_100_3.pt')
            ),
        ),
        Proportional_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_5k_3.pt')
            ),
        ),
        Proportional_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_1k_3.pt')
            ),
        ),
        Proportional_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_500_3.pt')
            ),
        ),
        Proportional_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_250_3.pt')
            ),
        ),
        Proportional_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_100_3.pt')
                    ),
        ),
        Movebase_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_5k_3.pt')
            ),
        ),
        Movebase_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_1k_3.pt')
            ),
        ),
        Movebase_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_500_3.pt')
            ),
        ),
        Movebase_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_250_3.pt')
            ),
        ),
        Movebase_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_100_3.pt')
            ),
        ),
    ),
    "LoCoBot-Lite": RobotNoiseModel(
        ILQR_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_5k_3.pt')
            ),
        ),
        ILQR_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_1k_3.pt')
            ),
        ),
        ILQR_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_500_3.pt')
            ),
        ),
        ILQR_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_250_3.pt')
            ),
        ),
        ILQR_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('ilqr_1.0_100_3.pt')
            ),
        ),
        Proportional_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_5k_3.pt')
            ),
        ),
        Proportional_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_1k_3.pt')
            ),
        ),
        Proportional_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_500_3.pt')
            ),
        ),
        Proportional_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_250_3.pt')
            ),
        ),
        Proportional_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('proportional_1.0_100_3.pt')
                    ),
        ),
        Movebase_5k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_5k_3.pt')
            ),
        ),
        Movebase_1k=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_1k_3.pt')
            ),
        ),
        Movebase_500=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_500_3.pt')
            ),
        ),
        Movebase_250=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_250_3.pt')
            ),
        ),
        Movebase_100=ControllerNoiseModel(
            motion_noise=MotionNoiseModel(
                _ActuationModel('movebase_1.0_100_3.pt')
            ),
        ),
    ),
}


@attr.s(auto_attribs=True)
class RegressionActuationSpec(ActuationSpec):
    r"""Regression actuation NN Proportional 1.0 
    """
    robot: str = attr.ib(default="LoCoBot")

    @robot.validator
    def check(self, attribute, value):
        assert value in noise_models.keys(), f"{value} not a known robot"

    controller: str = attr.ib(default="ILQR")

    @controller.validator
    def check(self, attribute, value):
        assert value in [
            "ILQR_5k",
            "ILQR_1k",
            "ILQR_500",
            "ILQR_250",
            "ILQR_100",
            "Proportional_5k",
            "Proportional_1k",
            "Proportional_500",
            "Proportional_250",
            "Proportional_100",
            "Movebase_5k",
            "Movebase_1k",
            "Movebase_500",
            "Movebase_250",
            "Movebase_100",
            "ILQR",
            "Proportional",
            "Movebase",
        ], f"{value} not a known controller"

    noise_multiplier: float = 1.0


_X_AXIS = 0
_Y_AXIS = 1
_Z_AXIS = 2


def _noisy_action_impl(
    scene_node: hsim.SceneNode,
    translate_amount: float,
    rotate_amount: float,
    multiplier: float,
    model: MotionNoiseModel,
    motion_type: str,
):
    # Perform the action in the coordinate system of the node
    transform = scene_node.transformation
    move_ax = -transform[_Z_AXIS].xyz
    perp_ax = transform[_X_AXIS].xyz

    if motion_type == "rotational": 
        if rotate_amount < 0: # right
            action = [1, 0, 0]
        elif rotate_amount > 0:
            action = [0, 1, 0]
    if motion_type == "linear": 
        if translate_amount > 0:
            action = [0, 0, 1]
    position = np.array([scene_node.translation.x, scene_node.translation.y, scene_node.translation.z])
    rotation = quat_from_magnum(scene_node.rotation)

    motion_noise = model.motion.sample(position, rotation, action)
    scene_node.translate_local(
        move_ax * motion_noise[1] # y
        + perp_ax * motion_noise[0] # x
    )

    scene_node.rotate_y_local(mn.Rad(motion_noise[2]))
    scene_node.rotation = scene_node.rotation.normalized()

@registry.register_move_fn(body_action=True)
class RegressionMoveBackward(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: RegressionActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            -actuation_spec.amount,
            0.0,
            actuation_spec.noise_multiplier,
            noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].motion_noise,
            "linear",
        )


@registry.register_move_fn(body_action=True)
class RegressionMoveForward(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: RegressionActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            actuation_spec.amount,
            0.0,
            actuation_spec.noise_multiplier,
            noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].motion_noise,
            "linear",
        )


@registry.register_move_fn(body_action=True)
class RegressionTurnLeft(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: RegressionActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            0.0,
            actuation_spec.amount,
            actuation_spec.noise_multiplier,
            noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].motion_noise,
            "rotational",
        )


@registry.register_move_fn(body_action=True)
class RegressionTurnRight(SceneNodeControl):
    def __call__(
        self, scene_node: hsim.SceneNode, actuation_spec: RegressionActuationSpec
    ):
        _noisy_action_impl(
            scene_node,
            0.0,
            -actuation_spec.amount,
            actuation_spec.noise_multiplier,
            noise_models[actuation_spec.robot][
                actuation_spec.controller
            ].motion_noise,
            "rotational",
        )

