#!/bin/env python
import rtde_control
import rtde_receive
import time

# --- Configuration ---
ROBOT_IP = "127.0.0.1"  # Replace with your robot's IP
FREQUENCY = 500.0  # RTDE frequency (Hz)
DT = 1.0 / FREQUENCY # Period of each RTDE communication cycle

# --- 1. Basic Connection and Disconnection ---
try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP) 
    print("Successfully connected to RTDEControlInterface.")

    # You can also set flags in the constructor for specific behaviors
    # For example, to use the ExternalControl UR Cap:
    # rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP, FREQUENCY, rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP)

except Exception as e:
    print(f"Error connecting to RTDEControlInterface: {e}")
    exit()

finally:
    if rtde_c:
        rtde_c.disconnect()
        print("Disconnected from RTDEControlInterface.")

# --- 2. Move Linear (moveL) ---
# Moves the TCP linearly to a target pose.
# Pose format: [x, y, z, rx, ry, rz] in meters and rad (Rx, Ry, Rz are Euler angles or axis-angle representation)
# Speed: m/s, Acceleration: m/s^2
print("\n--- Example: Move Linear (moveL) ---")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
if rtde_c.isConnected():
    # Example target pose (adjust for your robot's workspace and safety)
    # This is a sample, ensure it's a safe and reachable pose for your robot.
    target_pose = [0.1, -0.4, 0.2, 0.0, 3.14, 0.0]  # x, y, z, rx, ry, rz
    speed = 0.2  # m/s
    acceleration = 0.5 # m/s^2

    print(f"Moving to pose: {target_pose}")
    rtde_c.moveL(target_pose, speed, acceleration)
    print("moveL command sent. Waiting for completion...")
    # moveL is blocking by default, meaning it waits for the movement to finish.
    # For non-blocking, you might use an async version or check `isProgramRunning()`

    print("Movement complete.")
    rtde_c.disconnect()

# --- 3. Move Joint (moveJ) ---
# Moves the robot to a target joint configuration.
# Joint positions: [q0, q1, q2, q3, q4, q5] in radians
# Speed: rad/s, Acceleration: rad/s^2
print("\n--- Example: Move Joint (moveJ) ---")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
if rtde_c.isConnected():
    # Example target joint positions (adjust for your robot's home/safe position)
    target_joints = [-1.57, -1.57, 0.0, -1.57, 0.0, 0.0] # All joints to -90 degrees, except J3 and J5
    speed_j = 0.5 # rad/s
    acceleration_j = 1.0 # rad/s^2

    print(f"Moving to joint positions: {target_joints}")
    rtde_c.moveJ(target_joints, speed_j, acceleration_j)
    print("moveJ command sent. Waiting for completion...")
    print("Movement complete.")
    rtde_c.disconnect()


# --- 4. Real-time Control with servoJ (Joint Servo Control) ---
# This is for sending continuous, high-frequency joint commands.
# Requires a `RTDEReceiveInterface` to get the current robot state for feedback.
print("\n--- Example: Real-time Servo Control (servoJ) ---")
rtde_c = None
rtde_r = None
try:
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP, FREQUENCY)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP, FREQUENCY)

    if not rtde_c.isConnected() or not rtde_r.isConnected():
        raise RuntimeError("Could not connect to robot interfaces.")

    # Get initial joint positions
    initial_q = rtde_r.getActualQ()
    print(f"Initial joint positions: {initial_q}")

    # Define a small joint offset to move back and forth
    offset = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0] # Move only joint 0
    target_q_1 = [q + o for q, o in zip(initial_q, offset)]
    target_q_2 = [q - o for q, o in zip(initial_q, offset)]

    # Servo control parameters
    speed_factor = 0.5
    lookahead_time = 0.1 # seconds, range [0.03, 0.2]
    gain = 300 # proportional gain, range [100, 2000]

    print("Starting servoJ loop...")
    for _ in range(5): # Loop 5 times back and forth
        # Move to target_q_1
        start_time = time.time()
        while time.time() - start_time < 2.0: # Move for 2 seconds
            current_q = rtde_r.getActualQ()
            if current_q is None:
                print("Failed to get current joint positions, stopping servoJ.")
                break
            # This calculates a simple linear interpolation towards the target
            # More sophisticated control would use PID or other algorithms
            command_q = [curr + (targ - curr) * speed_factor for curr, targ in zip(current_q, target_q_1)]
            rtde_c.servoJ(command_q, 0.0, 0.0, DT, lookahead_time, gain) # speed and acceleration are ignored in servoJ with time parameter
            time.sleep(DT)

        # Move to target_q_2
        start_time = time.time()
        while time.time() - start_time < 2.0: # Move for 2 seconds
            current_q = rtde_r.getActualQ()
            if current_q is None:
                print("Failed to get current joint positions, stopping servoJ.")
                break
            command_q = [curr + (targ - curr) * speed_factor for curr, targ in zip(current_q, target_q_2)]
            rtde_c.servoJ(command_q, 0.0, 0.0, DT, lookahead_time, gain)
            time.sleep(DT)

    rtde_c.servoStop() # Important to stop servoing
    print("servoJ loop finished and stopped.")

except Exception as e:
    print(f"Error during servoJ example: {e}")
finally:
    if rtde_c:
        rtde_c.disconnect()
    if rtde_r:
        rtde_r.disconnect()

# --- 5. Setting Speed Slider Fraction ---
# This adjusts the overall speed of movements. Value between 0.01 and 1.0.
print("\n--- Example: Set Speed Slider Fraction ---")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
if rtde_c.isConnected():
    print("Setting speed slider to 50%")
    #SSC: It appears setSppedSlider() is not available in Python API
    #? rtde_c.setSpeedSlider(0.5)
    time.sleep(1) # Give robot time to apply
    print("Setting speed slider back to 100%")
    #? rtde_c.setSpeedSlider(1.0)
    rtde_c.disconnect()

# --- 6. Force Mode (Advanced) ---
# Allows the robot to move according to external forces/torques or apply a specific force.
# This is a complex topic and requires careful configuration.
# The example here is illustrative and should be used with caution and understanding of force mode.
print("\n--- Example: Force Mode ---")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
if rtde_c.isConnected():
    print("Entering force mode (illustrative example, be careful!)")
    # Define the task frame (relative to base)
    task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Typically identity for base frame
    # Selection vector: [fx, fy, fz, tx, ty, tz] - 0 for free, 1 for controlled
    selection_vector = [0, 0, 1, 0, 0, 0] # Control force in Z direction
    # Force to apply/maintain [N or Nm]
    wrench_target = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0] # Apply -10N in Z (down)
    # Type of force mode: 1 (constant force), 2 (compliant to external), etc.
    # Refer to URScript documentation for details on force_mode types
    type = 2
    # Limits for compliance/speed
    limits = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # Max speeds/rotational speeds

    # You might want to use a thread for force mode to monitor and stop
    # This is a simplified, blocking example.
    rtde_c.forceMode(task_frame, selection_vector, wrench_target, type, limits)
    print("Force mode enabled for 5 seconds...")
    time.sleep(5)
    rtde_c.forceModeStop()
    print("Force mode stopped.")
    rtde_c.disconnect()

# --- 7. Jogging (Manual Control) ---
# Provides a way to move the robot manually from your script.
print("\n--- Example: Jogging ---")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
if rtde_c.isConnected():
    print("Starting joint jogging (joint 0 positive, joint 1 negative)...")
    # Speed to jog at for each joint (rad/s)
    # [q0_speed, q1_speed, ..., q5_speed]
    joint_speeds = [0.1, -0.1, 0.0, 0.0, 0.0, 0.0]
    #- SSC: rtde_c.jogStart(joint_speeds, 0.0) # Last arg is tool speed (unused for joint jog)
    rtde_c.jogStart(joint_speeds)
    time.sleep(3) # Jog for 3 seconds
    rtde_c.jogStop()
    print("Jogging stopped.")

    print("Starting Cartesian jogging (move in X positive)...")
    # Speed to jog at in Cartesian space (m/s, rad/s)
    # [x_speed, y_speed, z_speed, rx_speed, ry_speed, rz_speed]
    cartesian_speeds = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
    #SSC: Something wrong with the argument list
    # rtde_c.jogStart(0.0, cartesian_speeds) # First arg is joint speed (unused for cartesian jog)
    rtde_c.jogStart(cartesian_speeds) # First arg is joint speed (unused for cartesian jog)
    time.sleep(3) # Jog for 3 seconds
    rtde_c.jogStop()
    print("Cartesian jogging stopped.")
    rtde_c.disconnect()
