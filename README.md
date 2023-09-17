# Self-supervised Traffic Advisor 2
## Requirements
- Python 3.7.5
- CARLA
  - CARLA 0.9.11

## Setup
```
conda env create -f environment.yml
conda install scikit-geometry -c conda-forge
pip install pytope
git clone https://github.com/sisl/InteractionSimulator.git
```
Modify [_kind_from_str](https://github.com/rosshemsley/interactionviz/blob/032eef47667e0748f14cd27f675cbff1a0a1bf37/interactionviz/tracks/tracks.py#L95-L103) to be
```python
def _kind_from_str(agent_type: str) -> AgentKind:
    if agent_type in ["car", 'truck', 'bus']:
        return AgentKind.CAR
    elif agent_type == "pedestrian":
        return AgentKind.PEDESTRIAN
    elif agent_type in ["pedestrian/bicycle", 'motorcycle']:
        return AgentKind.BICYCLE
    else:
        raise ValueError(f"unknown agent type: {agent_type}")
```
Then
```bash
pip install interactionviz
```
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
```bash
conda install pyg -c pyg -c conda-forge
# Or
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
```

### Setup CARLA
```
Download CARLA 0.9.11 from https://github.com/carla-simulator/carla/releases
wget -c https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
HOME_DIR=$HOME
mkdir -p ${HOME_DIR}/App/CARLA_0.9.11/
mv CARLA_0.9.11.tar.gz ${HOME_DIR}/App/CARLA_0.9.11/
cd ${HOME_DIR}/App/CARLA_0.9.11/
tar xvf CARLA_0.9.11.tar.gz
```
Add the following variables to `~/.bashrc` or `~/.zshrc`:
```
export CARLA_ROOT=${HOME_DIR}/App/CARLA_0.9.11
export CARLA_SERVER=${HOME_DIR}/App/CARLA/CarlaUE4.sh
# ${CARLA_ROOT} is the CARLA installation directory
# ${SCENARIO_RUNNER} is the ScenarioRunner installation directory
# <VERSION> is the correct string for the Python version being used
# In a build from source, the .egg files may be in: ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```
Collect CARLA Data
```
cd ${CARLA_ROOT}  # ${HOME_DIR}/App/CARLA_0.9.11
./CarlaUE4.sh
cd ${Curr_DIR}  # Current SSTA code directory
ln -s /media/data/jack/Apps/CARLA_0.9.11/PythonAPI/carla .
cd carla_tools
python manual_control.py --autopilot
python spawn_npc.py -n 100 -w 0 --sync
# or manually record videos
python manual_control.py
```
Manual control without rendering
The script PythonAPI/examples/no_rendering_mode.py provides an overview of the simulation. It creates a minimalistic aerial view with Pygame, that will follow the ego vehicle. This could be used along with manual_control.py to generate a route with barely no cost, record it, and then play it back and exploit it to gather data.
```
cd /opt/carla/PythonAPI/examples
python3 manual_control.py
cd /opt/carla/PythonAPI/examples
python3 no_rendering_mode.py --no-rendering
# Press "i" toggle actor id
```
Compute T2NO/T2ND (generate training data):
```
cd tools
python background_subtraction.py  # T2NO(), T2ND(), cv2_tracking()
```

Train video prediction model
```
python video_pred.py
```

Planning and Control using CARLA
```
cd carla_tools
bash run_manual_control_autopilot.sh
```

### Utils
#### img2video
```
cd tools
python background_subtraction.py (img2video())
# Or
python img2video.py (img2video())
```

## Related Work
- torchbeast_v3: Distributed Training, [TorchBeast](https://github.com/facebookresearch/torchbeast) is a distributed RL library 

# Reference
**[Connected Autonomous Vehicle Motion Planning with Video Predictions from Smart, Self-Supervised Infrastructure](https://arxiv.org/abs/2309.07504)**
<br />
[Jiankai Sun](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en),
[Shreyas Kousik](https://www.shreyaskousik.com/), 
[David Fridovich-Keil](https://dfridovi.github.io/), and
[Mac Schwager](http://web.stanford.edu/~schwager/)
<br />
**In IEEE 26th International Conference on Intelligent Transportation Systems (ITSC) 2023**
<br />
[[Paper]](https://arxiv.org/abs/2309.07504)[[Code]](https://github.com/Jiankai-Sun/SSTA2-ITSC-2023)

```
@ARTICLE{sun2023connected,
     author={J. {Sun} and S. {Kousik} and D. {Fridovich-Keil} and M. {Schwager}},
     journal={IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)},
     title={Connected Autonomous Vehicle Motion Planning with Video Predictions from Smart, Self-Supervised Infrastructure},
     year={2023},
}
```