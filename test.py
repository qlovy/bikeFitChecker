# sources bike fit body angle: https://www.bikefittr.com/bike-fit-angle-chart

from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO26 pose model
model = YOLO("yolo26n-pose.pt")

theory_angle = {
    "knee": [140, 150],
    "hip": [45, 60],
    "elbow": [150, 170],
    "back": [35, 50]
}

def run_keypoint_on_video(video_path, output_path="output_video.mp4", conf=0.5):
    keypoint_name = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    ]

    """Run YOLO26 keypoint estimation on a video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing : {video_path}")
    print(f"Resolution : {width}x{height} @ {fps}fps | {total} total frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    dict_result = {}
    angle_results = {}

    left_index = {
        "knee": keypoint_name.index("Left Knee"),
        "hip": keypoint_name.index("Left Hip"),
        "shoulder": keypoint_name.index("Left Shoulder"),
        "elbow": keypoint_name.index("Left Elbow"),
        "wrist": keypoint_name.index("Left Wrist"),
        "ankle": keypoint_name.index("Left Ankle")
    }

    max_knee_angle = 0
    min_hip_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)

        xy_val = results[0].keypoints.xy[0]

        left_xy = {
            "knee": xy_val[left_index["knee"]].numpy(),
            "hip": xy_val[left_index["hip"]].numpy(),
            "shoulder": xy_val[left_index["shoulder"]].numpy(),
            "elbow": xy_val[left_index["elbow"]].numpy(),
            "wrist": xy_val[left_index["wrist"]].numpy(),
            "ankle": xy_val[left_index["ankle"]].numpy()
        }

        knee_angle = get_angle_between(left_xy["ankle"], left_xy["knee"], left_xy["hip"])
        hip_angle = get_angle_between(left_xy["knee"], left_xy["hip"], left_xy["shoulder"])

        if frame_count == 0:
            angle_results = {
                "knee": knee_angle,
                "hip": hip_angle,
                "elbow": get_angle_between(left_xy["shoulder"], left_xy["elbow"], left_xy["wrist"]),
                "back": get_angle_between(left_xy["shoulder"], left_xy["hip"], np.array([left_xy["shoulder"][0], left_xy["hip"][1]]) )
            }
            max_knee_angle = knee_angle
            min_hip_angle = hip_angle

        # knee angle is measured at the bottom of the pedal stroke, so max value
        max_knee_angle = knee_angle if (max_knee_angle < knee_angle) else max_knee_angle

        # hip angle is measures at the top of the pedal stroke, so min value
        min_hip_angle = hip_angle if (min_hip_angle > hip_angle) else min_hip_angle

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total} frames...")

        if frame_count == total:
            angle_results["knee"] = max_knee_angle
            angle_results["hip"] = min_hip_angle

    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")
    return angle_results


def get_angle_between(p1, p2, p3):
    u1 = p1 - p2
    u2 = p3 - p2
    dot_prod = np.dot(u1, u2)
    mag_u1 = np.linalg.norm(u1)
    mag_u2 = np.linalg.norm(u2)
    angle = np.degrees( np.arccos( dot_prod / (mag_u1 * mag_u2) ) )
    return angle

def compare_angle(angle, arr_angle):
    return arr_angle[0] <= angle <= arr_angle[1]

def print_result(dict1, dict2):
    print("Results from video analytics: ")
    for key in dict1:
        print(f"  {key} angle = {dict1[key]: .1f}   Interval = {dict2[key][0]} - {dict2[key][1]}   Valid = {compare_angle(dict1[key], theory_angle[key])}")

measure_angles = run_keypoint_on_video("./video/IMG_0385.MP4", output_path="./output/output.mp4")

if measure_angles is None:
    print("Issue with video analytics")
    exit()

print_result(measure_angles, theory_angle)