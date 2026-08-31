from ultralytics import YOLO
import cv2

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        if frame_count == 0:
            for _, kps in enumerate(results[0].keypoints.xy):
                for i, (x, y) in enumerate(kps):
                    dict_result[f"{KEYPOINT_NAMES[i]}"] = (x, y)



        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total} frames...")

    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")


run_keypoint_on_video("./video/IMG_0385.MP4", output_path="./output/output.mp4")