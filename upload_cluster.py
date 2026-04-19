from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os


def upload_through_youtube_api(clusters: list, prefix="cluster"):
    SCOPES = ["https://www.googleapis.com/auth/youtube"]

    secret_path = r"PlaylistGenreClassification/client_secret_.json"
    flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
    credentials = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    youtube = build("youtube", "v3", credentials=credentials)

    def create_playlist(title, description=""):
        request = youtube.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {"title": title, "description": description},
                "status": {"privacyStatus": "private"},
            },
        )
        response = request.execute()
        return response["id"]

    def add_video_to_playlist(playlist_id, video_id):
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()

    for c in clusters:
        playlist_id = create_playlist(f"{prefix} - {c}")
        song_files = os.listdir(os.path.join(clusters_dir, c))
        for s in song_files:
            # extract id file path
            id = s.split(" -.- ")[1]
            # upload
            add_video_to_playlist(playlist_id, id)
        print(f"Finished uploading to playlist {c} with {len(song_files)} songs")


def get_manual_playlist_link(song_ids: list):
    def chunk(data, size):
        for i in range(0, len(data), size):
            yield data[i : i + size]

    for i, group in enumerate(chunk(song_ids, size=50)):
        print(f" ----- URL NUMBER {i} -----")
        url = "https://www.youtube.com/watch_videos?video_ids=" + ",".join(group)
        print(url)


cluster_prefix = "all time favs playlist"
clusters_dir = "clusters"
clusters = os.listdir(clusters_dir)

# clusters = [clusters[0]]  # for debugging

# API method:
upload_through_youtube_api(clusters, cluster_prefix)

# manual method:
for c in clusters:
    song_files = os.listdir(os.path.join(clusters_dir, c))
    song_ids = [s.split(" -.- ")[1] for s in song_files]
    print("URLs for chunk", c)
    get_manual_playlist_link(song_ids)
    print(f"End of chunks for {c}. Total size: {len(song_ids)}")
