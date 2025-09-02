PANDA_DEFAULT_CFG = {
    "panda_joint1": 0.040505826194643726,
    "panda_joint2": -0.00536819607860028,#
    "panda_joint3": -0.18538284760651613,
    "panda_joint4": -2.217976190474148,
    "panda_joint5": -0.005728349209802673,
    "panda_joint6": 2.223454317248339,
    "panda_joint7": 0.6904711141123535,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
FR3_DEFAULT_CFG = {
    "fr3_joint1": 0.03995192,
    "fr3_joint2": -0.00592799,#
    "fr3_joint3": -0.18448183,
    "fr3_joint4": -2.21950033,
    "fr3_joint5": -0.00492253,
    "fr3_joint6": 2.22304146,
    "fr3_joint7": 0.69084185,
    "fr3_finger_joint1": 0.04,
    "fr3_finger_joint2": 0.04,
}
XML_DEFAULT_CFG = {
    "joint1": 0.03995192,
    "joint2": -0.00592799,#
    "joint3": -0.18448183,
    "joint4": -2.21950033,
    "joint5": -0.00492253,
    "joint6": 2.22304146,
    "joint7": 0.69084185,
    "finger_joint1": 0.04,
    "finger_joint2": 0.04,
}
view_franka_joint = [0.03995192, -0.00592799, -0.18448183, -2.21950033, -0.00492253,  2.22304146,
  0.69084185]

BEST_PARAMS = {
    "kp": [
        2424.657254246728,
        16976.159595959594,
        12101.932123237386,
        10031.517126148701,
        1238.0951824004453,
        69.25030758714965,
        396.801903749272
    ],
    "kv": [
        2003.6817042606517,
        3271.0,
        1537.591478696742,
        1456.187969924812,
        460.44862155388466,
        13.506265664160395,
        80.49874686716791
    ],
}

CAMERA_INTRINSICS_DIR = "assets/realsense/cam_intr.json"
CAMERA_EXTRINSICS_DIR = "assets/realsense/cam_extr_init.txt"
OBJECT_URDF = {
    "gold": "outputs/gold-202504041202/object.urdf",
    "wood": "outputs/wood-20250409_223302_621/object.urdf",
    "box": "outputs/box-202504041230/object.urdf",
    "oreo": "outputs/box-202504192344_oreo/object.urdf",
    "ovaltine": "outputs/box-202504192344_ovaltine/object.urdf",
    "milk": "outputs/milk-box-20250827_180907_055/object.urdf",
    "banana": "outputs/banana-20250813_150647_221/object.urdf",
    "oreo": "outputs/oreo-box-20250830_141326_886/object.urdf",
}