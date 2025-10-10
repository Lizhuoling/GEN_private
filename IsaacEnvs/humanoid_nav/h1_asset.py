import isaaclab.sim as sim_utils
from basic_locomotion_dls_isaaclab.actuators import IdentifiedActuatorElectricCfg
from isaaclab.assets.articulation import ArticulationCfg


# -------------------------- 1.  actuator配置（按关节类型分组）--------------------------
# 从URDF提取的关键参数：effort_limit（力矩限制）、velocity_limit（速度限制）
# 同一类型关节（如髋关节偏航）共享相同参数，用正则表达式匹配关节名

# 1.1 腿部关节配置（左/右髋、膝、踝）
H1_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_hip_yaw_joint"],  # 匹配左/右髋偏航关节
    effort_limit=200.0,  # 从URDF <limit effort="200">提取
    velocity_limit=23.0,  # 从URDF <limit velocity="23">提取
    saturation_effort=200.0,  # 饱和力矩与限制一致
    stiffness=25.0,  # 参考Aliengo，可根据实际调试调整
    damping=2.0,     # 参考Aliengo，可根据实际调试调整
    armature=0.01,   # 电枢惯性，参考Aliengo
    friction_static=0.2,  # 静摩擦，参考Aliengo
    activation_vel=0.1,   # 激活速度阈值，参考Aliengo
    friction_dynamic=0.6, # 动摩擦，参考Aliengo
)

H1_HIP_ROLL_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_hip_roll_joint"],  # 匹配左/右髋侧倾关节
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=25.0,
    damping=2.0,
    armature=0.01,
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.6,
)

H1_HIP_PITCH_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_hip_pitch_joint"],  # 匹配左/右髋俯仰关节
    effort_limit=200.0,
    velocity_limit=23.0,
    saturation_effort=200.0,
    stiffness=25.0,
    damping=2.0,
    armature=0.01,
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.6,
)

H1_KNEE_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_knee_joint"],  # 匹配左/右膝关节
    effort_limit=300.0,  # 从URDF <limit effort="300">提取
    velocity_limit=14.0,  # 从URDF <limit velocity="14">提取
    saturation_effort=300.0,
    stiffness=25.0,
    damping=2.0,
    armature=0.01,
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.6,
)

H1_ANKLE_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_ankle_joint"],  # 匹配左/右踝关节
    effort_limit=40.0,  # 从URDF <limit effort="40">提取
    velocity_limit=9.0,  # 从URDF <limit velocity="9">提取
    saturation_effort=40.0,
    stiffness=25.0,
    damping=2.0,
    armature=0.01,
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.6,
)

# 1.2 躯干关节配置
H1_TORSO_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=["torso_joint"],  # 躯干旋转关节
    effort_limit=200.0,  # 从URDF <limit effort="200">提取
    velocity_limit=23.0,  # 从URDF <limit velocity="23">提取
    saturation_effort=200.0,
    stiffness=25.0,
    damping=2.0,
    armature=0.01,
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.6,
)

# 1.3 手臂关节配置（左/右肩、肘）
H1_SHOULDER_PITCH_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_shoulder_pitch_joint"],  # 匹配左/右肩俯仰关节
    effort_limit=40.0,  # 从URDF <limit effort="40">提取
    velocity_limit=9.0,  # 从URDF <limit velocity="9">提取
    saturation_effort=40.0,
    stiffness=20.0,  # 手臂负载较小，可适当降低刚度（参考值）
    damping=1.5,     # 对应降低阻尼（参考值）
    armature=0.005,  # 手臂惯性较小，电枢惯性降低（参考值）
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.3,
)

H1_SHOULDER_ROLL_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_shoulder_roll_joint"],  # 匹配左/右肩侧倾关节
    effort_limit=40.0,
    velocity_limit=9.0,
    saturation_effort=40.0,
    stiffness=20.0,
    damping=1.5,
    armature=0.005,
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.3,
)

H1_SHOULDER_YAW_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_shoulder_yaw_joint"],  # 匹配左/右肩偏航关节
    effort_limit=18.0,  # 从URDF <limit effort="18">提取
    velocity_limit=20.0,  # 从URDF <limit velocity="20">提取
    saturation_effort=18.0,
    stiffness=20.0,
    damping=1.5,
    armature=0.005,
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.3,
)

H1_ELBOW_ACTUATOR_CFG = IdentifiedActuatorElectricCfg(
    joint_names_expr=[".*_elbow_joint"],  # 匹配左/右肘关节
    effort_limit=18.0,  # 从URDF <limit effort="18">提取
    velocity_limit=20.0,  # 从URDF <limit velocity="20">提取
    saturation_effort=18.0,
    stiffness=20.0,
    damping=1.5,
    armature=0.005,
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.3,
)


# -------------------------- 2. 核心配置（ArticulationCfg）--------------------------
H1_CFG = ArticulationCfg(
    prim_path=None,  # 运行时动态指定URDF路径，无需提前设置
    # -------------------------- 2.1 USD文件加载配置
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"h1_asset/h1.usd",  # 替换为你的H1 USD文件实际路径
        activate_contact_sensors=True,  # 启用接触传感器（用于步态规划/碰撞检测）
        # 刚体物理属性（从URDF惯性参数映射，参考Aliengo调整）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 启用重力
            retain_accelerations=False,  # 不保留加速度（默认）
            linear_damping=0.0,  # 线性阻尼（URDF无，设为0）
            angular_damping=0.0,  # 角阻尼（URDF无，设为0）
            max_linear_velocity=1000.0,  # 最大线速度（避免数值溢出）
            max_angular_velocity=1000.0,  # 最大角速度（避免数值溢出）
            max_depenetration_velocity=1.0,  # 最大解穿透速度（防止碰撞抖动）
        ),
        # 关节链物理属性（影响仿真稳定性）
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # 启用自碰撞（避免肢体穿透）
            solver_position_iteration_count=4,  # 位置求解器迭代次数（平衡精度与速度）
            solver_velocity_iteration_count=0,  # 速度求解器迭代次数（默认0）
        ),
    ),
    # -------------------------- 2.2 初始状态配置（站立姿态，参考URDF关节限位）
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-6.0, -1.0, 1.0),
        rot=(0.0, 0.0, 0.0, 1.0),   # (w, x, y, z)
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
    # -------------------------- 2.3 关联所有Actuator配置
    actuators={
        # 腿部
        "hip_yaw": H1_HIP_YAW_ACTUATOR_CFG,
        "hip_roll": H1_HIP_ROLL_ACTUATOR_CFG,
        "hip_pitch": H1_HIP_PITCH_ACTUATOR_CFG,
        "knee": H1_KNEE_ACTUATOR_CFG,
        "ankle": H1_ANKLE_ACTUATOR_CFG,
        # 躯干
        "torso": H1_TORSO_ACTUATOR_CFG,
        # 手臂
        "shoulder_pitch": H1_SHOULDER_PITCH_ACTUATOR_CFG,
        "shoulder_roll": H1_SHOULDER_ROLL_ACTUATOR_CFG,
        "shoulder_yaw": H1_SHOULDER_YAW_ACTUATOR_CFG,
        "elbow": H1_ELBOW_ACTUATOR_CFG,
    },
    # 关节限位安全系数（避免触碰URDF硬限位，0.95表示使用95%的限位范围）
    soft_joint_pos_limit_factor=0.95,
)