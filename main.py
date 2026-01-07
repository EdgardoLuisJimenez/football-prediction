from utils import read_videos, save_video
import cv2
from tracker import Tracker


def main():
    # read videos
    video_frames = read_videos("input_videos/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Save cropped image of a player
    for tracker_id, player in tracks['players'][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]

        # Crop bbox from frame
        # Result: A smaller images containing just the player 
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save the cropped image
        cv2.imwrite(f"output_videos/img/cropped_img.jpg", cropped_image)
        break


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save videos
    save_video(output_video_frames, "outputs/outputs_video/output_video.mp4")


if __name__ == "__main__":
    main()
