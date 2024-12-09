import sys
sys.path.append("../config")

import numpy as np
from autolab_core import RigidTransform
from robot_config import RobotConfig
from task_config import TaskConfig
import utils as utils
from robot import Robot
from frankapy import FrankaArm
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
import rospy


class TrajectoryGenerator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.max_vel = RobotConfig.MAX_VELOCITY
        self.max_acc = RobotConfig.MAX_ACCELERATION

    def quick_slerp(self, rotm1, rotm2, t, num_points):
        k = t/num_points
        q1 = utils._rotation_to_quaternion(self, rotm1)
        q2 = utils._rotation_to_quaternion(self, rotm2)
        return(utils._quaternion_to_rotation(utils._slerp(q1, q2, k)))
    
    def bezier(self, start, control, end, t, num_points):
        k = t/num_points
        x = ((1-k)**2)*start[0] + 2*(1-k)*k*control[0] + (k**2)*end
        y = ((1-k)**2)*start[1] + 2*(1-k)*k*control[0] + (k**2)*end
        z = ((1-k)**2)*start[2] + 2*(1-k)*k*control[2] + (k**2)*end
        return np.array([x,y,z])
    
    def generate_straight_line(self, start:np.array, end:np.array) -> np.array:
        """
        This function creates a smooth straight-line trajectory in Cartesian space.

         Parameters
        ----------
        start and end should be 4x4 FK matrices
            
        Return
        ------
        array_like
            Input to either interpolate_cartesian_trajectory() or convert_cartesian_to_joint()

        Raises
        ------
        NotImplementedError
            This function needs to be implemented.

        Hints
        -----
        - Need start pose (4x4 matrix) and end pose (4x4 matrix)
        - Use linear interpolation for position: p(t) = p0 + t*(p1-p0)
        - Use SLERP (spherical linear interpolation) for rotation
        - Number of points should give enough resolution for smooth motion
        - Each waypoint should be a 4x4 transformation matrix

        """
        start_pos = start[3, 0:3]
        start_r = start[0:3, 0:3]
        end_pos = end[3, 0:3]
        end_r = end[0:3, 0:3]

        num_points = np.linalg.norm(start_pos - end_pos) // TaskConfig.PATH_RESOLUTION*10
        poses = np.zeros(4, 4, num_points)
        for t in range(num_points):
            poses[3, 0:3, t] = start_pos + t*(end_pos-start_pos)
            poses[0:3, 0:3, t] = self.quick_slerp(start_r, end_r, t, num_points)
            poses[3, 3, t] = 1
        return poses


        raise NotImplementedError("Implement generate_straight_line")
        
    def generate_curve(self, start, end, control):
        """
        This function creates a smooth curved trajectory in Cartesian space.

        Parameters
        ----------
        You can define any parameters you need for this function.

        start and end should be 4x4 FK matrices
        roc is a float
        normal is normal of the plane
            
        Return
        ------
        array_like
            Input to either interpolate_cartesian_trajectory() or convert_cartesian_to_joint()

        Raises
        ------
        NotImplementedError
            This function needs to be implemented.

        Hints
        -----
        - Need list of points defining the curve
        - Can break curve into segments and use linear interpolation for each
        - Each waypoint is a 4x4 transformation matrix
        - Keep orientation aligned with curve direction
        - PATH_RESOLUTION from TaskConfig helps determine point spacing
        - Line segments should be small enough for smooth motion

        """

        start_pos = start[3, 0:3]
        start_r = start[0:3, 0:3]
        end_pos = end[3, 0:3]
        end_r = end[0:3, 0:3]
        control_pos = control[3, 0:3]

        num_points = np.linalg.norm(start_pos - end_pos) // TaskConfig.PATH_RESOLUTION
        poses = np.zeros(4, 4, num_points)

        for t in range(num_points):
            poses[3, 0:3, t] = self.bezier(start_pos, end_pos, control_pos, t, num_points)
            poses[0:3, 0:3, t] = self.quick_slerp(start_r, end_r, t, num_points)
            poses[3, 3, t] = 1

        return poses  

        raise NotImplementedError("Implement generate_curve")
    
    def interpolate_cartesian_trajectory(self, cartesian_trajectory):
        """
        Time-parameterize Cartesian trajectory with trapezoidal velocity profile.

        Parameters
        ----------
        cartesian_trajectory : list of np.ndarray
            List of 4x4 homogeneous transformation matrices representing the path in Cartesian space.
        
        Returns
        -------
        time_param_trajectory : list of np.ndarray
            List of 4x4 homogeneous transformation matrices representing the time-parameterized trajectory.

        Raises
        ------
        NotImplementedError
            This function needs to be implemented.

        Hints
        -----
        Key Requirements:  
        - Timing: Waypoints must be spaced exactly 20ms apart for controller
        - Safety: Stay within MAX_VELOCITY and MAX_ACCELERATION limits
        - Smoothness: Use trapezoidal velocity profile for acceleration/deceleration

        Implementation:
        - Calculate duration based on path length and velocity limits
        - Generate trapezoidal velocity profile with acceleration limits 
        - Ensure 20ms spacing between waypoints
        - For rotations: Use SLERP to interpolate between orientations
        """
        # Extract positions and orientations from the input homogeneous matrices
        positions = []
        orientations = []
        distances = []
        total_distance = 0.0
        
        for i in range(1, cartesian_trajectory.shape[2]):
            p1 = cartesian_trajectory[:3, 3, i-1]
            p2 = cartesian_trajectory[:3, 3, i]
            distance = np.linalg.norm(p2 - p1)
            total_distance += distance
            distances.append(distance)
            
            positions.append(p1)
            # Extract rotations from the top-left 3x3 block

            r = cartesian_trajectory[:3, :3, i-1]
            orientations.append((r))

        # Determine the time required based on velocity limits
        max_velocity = RobotConfig.MAX_VELOCITY
        max_acceleration = RobotConfig.MAX_ACCELERATION
        
        # Time to accelerate to max velocity
        time_to_max_velocity = max_velocity / max_acceleration
        
        # Distance covered during acceleration and deceleration (assuming symmetry)
        distance_acceleration = 0.5 * max_acceleration * time_to_max_velocity**2
        
        # Time to cover the entire trajectory with trapezoidal velocity profile
        if total_distance <= 2 * distance_acceleration:
            # The trajectory is short enough to only use acceleration and deceleration phases
            time_required = np.sqrt(2 * total_distance / max_acceleration)
        else:
            # Use acceleration, constant velocity, and deceleration phases
            constant_velocity_distance = total_distance - 2 * distance_acceleration
            time_at_constant_velocity = constant_velocity_distance / max_velocity
            time_required = 2 * time_to_max_velocity + time_at_constant_velocity
        
        # Generate time-parameterized trajectory (using 20ms spacing)
        num_points = int(np.ceil(time_required / 0.02))  # 20ms spacing
        times = np.linspace(0, time_required, num_points)
        
        # Calculate the velocity profile (trapezoidal)
        velocities = []
        for t in times:
            if t < time_to_max_velocity:
                velocity = max_acceleration * t
            elif t < time_required - time_to_max_velocity:
                velocity = max_velocity
            else:
                velocity = max_velocity - max_acceleration * (t - (time_required - time_to_max_velocity))
            velocities.append(velocity)
        
        # Interpolate positions and orientations using the velocity profile
        current_time = 0.0
        interpolated_matrices = []

        for i in range(1, len(cartesian_trajectory)):
            p1 = cartesian_trajectory[i-1][:3, 3]
            p2 = cartesian_trajectory[i][:3, 3]
            segment_distance = distances[i-1]
            
            # Linear interpolation of position based on the velocity profile
            while current_time < times[-1] and len(interpolated_matrices) < num_points:
                travel_distance = np.interp(current_time, times, velocities) * (current_time - times[0])
                travel_fraction = travel_distance / segment_distance
                position = p1 + travel_fraction * (p2 - p1)
                
                # Interpolate the orientation using SLERP
                r1, r2 = orientations[i-1], orientations[i]
                rotation_matrix = self.quick_slerp(r1, r2, current_time, times[-1])
                
                # Construct the 4x4 homogeneous transformation matrix
                homogenous_matrix = np.eye(4)
                homogenous_matrix[:3, :3] = rotation_matrix
                homogenous_matrix[:3, 3] = position
                homogenous_matrix[3, 3] = position
                
                interpolated_matrices.append(homogenous_matrix)
                current_time += 0.02  # 20ms step
        
        return interpolated_matrices

        
    def interpolate_joint_trajectory(self, joint_trajectory):
        """
        Time-parameterize joint trajectory with trapezoidal velocity profile.

        Parameters
        ----------
        joint_trajectory : array_like 
            Array of joint angles

        Returns
        -------
        array_like
            Time-parameterized trajectory with 20ms spacing

        Raises
        ------
        NotImplementedError
            This function needs to be implemented.

        Hints
        -----
        Key Requirements:
        - Timing: Waypoints must be spaced exactly 20ms apart for controller
        - Safety: Stay within MAX_VELOCITY and MAX_ACCELERATION limits 
        - Smoothness: Use trapezoidal velocity profile for acceleration/deceleration

        Implementation:
        - Use max velocity and acceleration from RobotConfig
        - Ensure smooth acceleration and deceleration
        - Keep 20ms between waypoints as required by controller

        """


    
    def convert_cartesian_to_joint(self, cartesian_trajectory):
        """
        Convert Cartesian trajectory to joint trajectory using inverse kinematics.

        Parameters
        ----------
        cartesian_trajectory : array_like
            Array of poses in Cartesian space

        Returns
        -------
        array_like
            Joint space trajectory

        Raises
        ------
        NotImplementedError
            This function needs to be implemented.

        Hints
        -----
        Key Requirements:
        - Safety: All solutions must respect joint limits
        - Smoothness: Solutions should minimize joint motion between waypoints

        Implementation:
        - Use Jacobian pseudo-inverse method  
        - Check joint limits after IK
        - Use previous joint solution as seed for next IK
        """
        trajectory = np.zeros([7, cartesian_trajectory])
        robot = Robot()
        trajectory[:, 0] = robot._inverse_kinematics(robot.get_joints(), cartesian_trajectory[:,:,0])
        for i in range(1, cartesian_trajectory[2]):
            trajectory[:, i] = robot._inverse_kinematics(trajectory[:, i-1],cartesian_trajectory[:,:,i])
        return trajectory




        raise NotImplementedError("Implement convert_cartesian_to_joint")

class TrajectoryFollower:
    def __init__(self):
        self.dt = 0.02  # Required 20ms control loop
        self.fa = FrankaArm()
        
    def follow_joint_trajectory(self, joint_trajectory):
        """
        Follow a joint trajectory using dynamic control.
        
        From writeup: Must have 20ms between waypoints and maintain smooth motion
        
        Parameters
        ----------
        joint_trajectory : np.ndarray
            Array of shape (N, 7) containing joint angles for each timestep
        """
        rospy.loginfo('Initializing Sensor Publisher')
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        rate = rospy.Rate(1 / self.dt)

        rospy.loginfo('Publishing joints trajectory...')
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_joints(joint_trajectory[0], duration=1000, dynamic=True, buffer_time=10)
        init_time = rospy.Time.now().to_time()
        for i in range(1, joint_trajectory.shape[0]):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=joint_trajectory[i]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            
            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            pub.publish(ros_msg)
            rate.sleep()

        # Stop the skill
        # Alternatively can call fa.stop_skill()
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        pub.publish(ros_msg)

        rospy.loginfo('Done')
