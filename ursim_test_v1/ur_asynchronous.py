#!/usr/bin/env python3
import rtde_control
import rtde_receive
import time
import json
import argparse
import sys


def main(robot_ip, json_file, acceleration,responsiveness):
    """
    """
    # Connect to UR via RTDE
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    if not (rtde_c.isConnected() and rtde_r.isConnected()):
        print("ERROR: RTDE connection failed.", file=sys.stderr)
        sys.exit(1)
    print(f"Connected to UR at {robot_ip}")

    # Open JSONL file and seek to end
    try:
        f = open(json_file, 'r')
    except FileNotFoundError:
        print(f"ERROR: File {json_file} not found.", file=sys.stderr)
        sys.exit(1)
    f.seek(0, 2)

    current_vel = [0.0]*6

    try:
        while True:
            # Read any new lines
            lines = f.readlines()
            if lines:
                # parse only the last appended entry
                try:
                    msg = json.loads(lines[-1])
                    current_vel = [
                        float(msg.get('dx', 0.0)),
                        float(msg.get('dy', 0.0)),
                        float(msg.get('dz', 0.0)),
                        float(msg.get('drx', 0.0)),
                        float(msg.get('dry', 0.0)),
                        float(msg.get('drz', 0.0))
                    ]
                except (ValueError, KeyError) as e:
                    print(f"Invalid JSON: {e}", file=sys.stderr)
            # Execute velocity command for 1 second
            print(f"[{time.strftime('%H:%M:%S')}] Applying vel: {current_vel}")
            rtde_c.speedL(current_vel, acceleration, responsiveness)
            time.sleep(responsiveness)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        f.close()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("Clean exit.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read tail of JSONL and apply delta velocities each second'
    )
    parser.add_argument('--robot-ip', default='127.0.0.1', help='UR IP')
    parser.add_argument('--json-file', default='asynchronous_deltas.jsonl', help='source jsonl file')
    parser.add_argument('--acceleration', type=float, default=0.5, help='Acceleration')
    parser.add_argument('--responsiveness', type=float, default = 1.0, help='responsiveness')
    args = parser.parse_args()
    main(args.robot_ip, args.json_file, args.acceleration, args.responsiveness)
