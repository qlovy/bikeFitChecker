from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO26 pose model
model = YOLO("yolo26n-pose.pt")


def run_keypoint_on_video(video_path, output_path="output_video.mp4", conf=0.5):
    KEYPOINT_NAMES = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    ]

    """Run YOLO26 keypoint estimation on a video file frame by frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        if frame_count == 0:
            for _, kps in enumerate(results[0].keypoints.xy):
                for i, t in enumerate(kps):
                    dict_result[f"{KEYPOINT_NAMES[i]}"] = t.numpy()


            angle_results["knee"] = get_angle_between(dict_result["Left Ankle"], dict_result["Left Knee"], dict_result["Left Hip"])
            angle_results["hip"] = get_angle_between(dict_result["Left Knee"], dict_result["Left Hip"], dict_result["Left Shoulder"])
            angle_results["elbow"] = get_angle_between(dict_result["Left Shoulder"], dict_result["Left Elbow"], dict_result["Left Wrist"])
            angle_results["back"] = get_angle_between(dict_result["Left Shoulder"], dict_result["Left Hip"], np.array([dict_result["Left Shoulder"][0], dict_result["Left Hip"][1]]) )

            cv2.imwrite("./img/frame0.jpg", results[0].plot())



        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total} frames...")

    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")
    print(angle_results)


def get_angle_between(p1, p2, p3):
    u1 = p1 - p2
    u2 = p3 - p2
    dot_prod = np.dot(u1, u2)
    mag_u1 = np.linalg.norm(u1)
    mag_u2 = np.linalg.norm(u2)
    angle = np.degrees( np.arccos( dot_prod / (mag_u1 * mag_u2) ) )
    return angle

run_keypoint_on_video("./video/IMG_0385.MP4", output_path="./output/output.mp4")