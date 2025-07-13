#!/bin/bash

# Polyscope UI is accessible from here:
# http://localhost:6080/vnc.html

docker run --rm -it \
  -p 5900:5900 \
  -p 6080:6080 \
  -p 29999:29999 \
  -p 30001-30004:30001-30004 \
  -e ROBOT_MODEL=UR5E \
  -e URSIM_ROBOT_MODE=SIMULATION \
  --name ursim_e_series \
  universalrobots/ursim_e-series

