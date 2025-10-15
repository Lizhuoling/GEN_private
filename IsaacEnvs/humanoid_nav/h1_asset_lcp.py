import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg

H1_HIP_YAW_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_yaw_joint"],
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=200.0,
    damping=5.0,
    armature=0.01,
)

H1_HIP_ROLL_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_roll_joint"],
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=200.0,
    damping=5.0,
    armature=0.01,
)

H1_HIP_PITCH_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_pitch_joint"],
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=200.0,
    damping=5.0,
    armature=0.01,
)

H1_KNEE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_knee_joint"],
    effort_limit=300.0,
    velocity_limit=14.0,
    saturation_effort=300.0,
    stiffness=200.0,
    damping=5.0,
    armature=0.01,
)

H1_ANKLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_ankle_joint"],
    effort_limit=40.0,
    velocity_limit=9.0,
    saturation_effort=40.0,
    stiffness=40.0,
    damping=2.0,
    armature=0.01,
)

H1_TORSO_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=["torso_joint"],
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=300.0,
    damping=6.0,
    armature=0.01,
)

H1_SHOULDER_PITCH_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_shoulder_pitch_joint"],
    effort_limit=40.0,
    velocity_limit=9.0,
    saturation_effort=40.0,
    stiffness=40.0,
    damping=2.0,
    armature=0.005,
)

H1_SHOULDER_ROLL_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_shoulder_roll_joint"],
    effort_limit=40.0,
    velocity_limit=9.0,
    saturation_effort=40.0,
    stiffness=40.0,
    damping=2.0,
    armature=0.005,
)

H1_SHOULDER_YAW_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_shoulder_yaw_joint"],
    effort_limit=18.0,
    velocity_limit=20.0,
    saturation_effort=18.0,
    stiffness=20.0,
    damping=1.5,
    armature=0.005,
)

H1_ELBOW_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*_elbow_joint"],
    effort_limit=18.0,
    velocity_limit=20.0,
    saturation_effort=18.0,
    stiffness=20.0,
    damping=1.5,
    armature=0.005,
)

H1_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"h1_asset/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-6.0, -1.0, 1.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.60,
            ".*_knee_joint": 1.20,
            ".*_ankle_joint": -0.60,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "hip_yaw": H1_HIP_YAW_ACTUATOR_CFG,
        "hip_roll": H1_HIP_ROLL_ACTUATOR_CFG,
        "hip_pitch": H1_HIP_PITCH_ACTUATOR_CFG,
        "knee": H1_KNEE_ACTUATOR_CFG,
        "ankle": H1_ANKLE_ACTUATOR_CFG,
        "torso": H1_TORSO_ACTUATOR_CFG,
        "shoulder_pitch": H1_SHOULDER_PITCH_ACTUATOR_CFG,
        "shoulder_roll": H1_SHOULDER_ROLL_ACTUATOR_CFG,
        "shoulder_yaw": H1_SHOULDER_YAW_ACTUATOR_CFG,
        "elbow": H1_ELBOW_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)