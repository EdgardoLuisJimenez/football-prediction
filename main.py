from utils import read_videos, save_video
import cv2
from team_assigner import TeamAssigner
from tracker import Tracker


def main():
    # read videos
    video_frames = read_videos("input_videos/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Assign Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track["bbox"], player_id)

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_color[team]


    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save videos
    save_video(output_video_frames, "outputs/outputs_video/output_video.mp4")


if __name__ == "__main__":
    main()
