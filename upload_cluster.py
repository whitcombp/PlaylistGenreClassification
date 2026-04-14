from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

SCOPES = ["https://www.googleapis.com/auth/youtube"]

secret_path = r"PlaylistGenreClassification/client_secret_.json"
flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
credentials = flow.run_local_server(host="localhost", port=8080, open_browser=False)

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


clusters_dir = "clusters"
clusters = os.listdir(clusters_dir)

clusters = [clusters[0]]  # for debugging

for c in clusters:
    playlist_id = create_playlist(c)
    song_files = os.listdir(os.path.join(clusters_dir, c))

    for s in song_files:
        # extract id file path
        id = s.split(" -.- ")[1]
        # upload
        add_video_to_playlist(playlist_id, id)
    print(f"Finished uploading to playlist {c} with {len(song_files)} songs")
