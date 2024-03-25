# SssssSLAM

This article presents SLAM with semantic segmentation, a functional monocular SLAM system operating on the basis of package configuration with ArUco initialization. The system is capable of not only creating camera tracking and mapping, but also using a Neural Network to identify stationary objects. Based on the algorithms of recent years, we have developed a new system that selects points and keyframes and creates a map reflecting the sequence of poses of the camera. You can see obtained results in the `results` folder.

It is a final project for the "Perception in Robotics" course at Skoltech, 2024.

## Reproducing the solution
Create an virtual environment:
```
cd "<project_dir>"
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
Activate the environment:
```
source venv/bin/activate
```
And run src/main.ipynb file. Done!
