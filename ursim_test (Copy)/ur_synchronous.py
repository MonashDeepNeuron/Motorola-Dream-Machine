#!/usr/bin/env python3
import rtde_control
import rtde_receive
import time
import json
import argparse
import sys

def main(robot_ip, json_source, json_log, speed, acceleration,responsiveness):
    """
    Stream JSON delta movements once per second as velocity commands and
    move the UR robot accordingly using RTDE.

    Each JSON line should contain dx, dy, dz (and optional drx, dry, drz).
    """
    # Establish RTDE connections
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    if not (rtde_c.isConnected() and rtde_r.isConnected()):
        print("ERROR: Could not connect to RTDE interfaces.", file=sys.stderr)
        sys.exit(1)
    print(f"Connected to UR at {robot_ip}")

    # Open JSON source (file or stdin) and log file
    in_f = sys.stdin if json_source == '-' else open(json_source, 'r')
    log_f = open(json_log, 'a')

    try:
        while True:
            # Read next JSON line
            line = in_f.readline()
            if not line:
                print("End of JSON source.")
                break

            # Parse JSON deltas
            try:
                msg = json.loads(line)
                dx  = float(msg.get('dx', 0.0))
                dy  = float(msg.get('dy', 0.0))
                dz  = float(msg.get('dz', 0.0))
                drx = float(msg.get('drx', 0.0))
                dry = float(msg.get('dry', 0.0))
                drz = float(msg.get('drz', 0.0))
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid JSON message: {e}", file=sys.stderr)
                continue

            # Retrieve current TCP pose (for logging)
            current_pose =   rtde_r.getActualTCPPose()
            target_pose = [
                current_pose[0] + dx,
                current_pose[1] + dy,
                current_pose[2] + dz,
                current_pose[3] + drx,
                current_pose[4] + dry,
                current_pose[5] + drz
            ]

            # Log timestamp, target, and delta
            timestamp = time.time()
            log_entry = {
                't': timestamp,
                'target_pose': target_pose,
                'delta': [dx, dy, dz, drx, dry, drz]
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

            # Execute 1-second Cartesian velocity command
            vel = [dx, dy, dz, drx, dry, drz]
            print(f"[{time.strftime('%H:%M:%S')}] Streaming velocity Δ/sec = {vel}")
            rtde_c.speedL(vel, acceleration, responsiveness)
            # speedL blocks for 1 second, ensuring fixed cadence
            time.sleep(responsiveness)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        in_f.close()
        log_f.close()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Disconnected from UR.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream JSON delta movements once per second and move the UR robot accordingly."
    )
    parser.add_argument(
        "--robot-ip", default="127.0.0.1",
        help="IP address of the UR robot or simulator"
    )
    parser.add_argument(
        "--json-source", default="synchronous_deltas.jsonl",
        help="Path to JSON-line file of {'dx','dy','dz',…} messages (use '-' for stdin)"
    )
    parser.add_argument(
        "--json-log", default="synchronous_log.jsonl",
        help="Path to append timestamped JSON log"
    )
    parser.add_argument(
        "--speed", type=float, default=0.2,
        help="Cartesian speed (unused with speedL, placeholder)"
    )
    parser.add_argument(
        "--acceleration", type=float, default=0.5,
        help="Acceleration for speedL (m/s²)"
    )
    parser.add_argument(
        "--responsiveness", type=float, default=1.0,
        help="How often each command line in json file takes effect (seconds)"
    )
    args = parser.parse_args()
    main(
        args.robot_ip,
        args.json_source,
        args.json_log,
        args.speed,
        args.acceleration,
        args.responsiveness
    )
