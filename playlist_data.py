from pytubefix import Playlist as PyfixPlaylist, YouTube
from pytubefix.exceptions import AgeRestrictedError
import os
from tqdm import tqdm


class Playlist:
    def __init__(self, url, download_path=None):
        self.p = PyfixPlaylist(url)
        # download playlist video data if path isn't empty
        if download_path is None:
            self.download_path = f"{self.p.title}_playlist"
        else:
            self.download_path = download_path
        os.makedirs(self.download_path, exist_ok=True)

        # store what's already downloaded to skip in-loop
        existing_songs = set(
            [p.split(" -.- ")[0] for p in os.listdir(self.download_path)]
        )
        # iterate through the songs that still need downloading and download them
        for url in tqdm(self.p.video_urls):
            # clean title for safe path
            video_id = url.split("v=")[-1]
            if video_id in existing_songs:  # skip
                continue
            else:  # download
                try:
                    yt = YouTube(url=url)
                    title = "".join(
                        c for c in yt.title if c not in r'\/:*?"<>|'
                    ).strip()
                    title = video_id + " -.- " + title + ".mp4"
                    stream = yt.streams.get_highest_resolution()
                    stream.download(output_path=self.download_path, filename=title)
                except AgeRestrictedError:
                    print("Age restricted error with title:", title)
                    continue

        self.v = [v for v in os.listdir(self.download_path)]

    def index(self, index):
        return self.v[index]

    def __len__(self):
        return len(self.v)


def main():
    playlist = Playlist(
        "https://www.youtube.com/playlist?list=PLwefRwBMRzc9HxYarR0EltiVQxNcdxAyd"
    )
    print(len(playlist))
    print(playlist.index(0))


if __name__ == "__main__":
    main()
