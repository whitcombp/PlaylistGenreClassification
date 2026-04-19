# Automatic AI Playlist Genre Organizer

This project uses contrastive learning and clustering to take YouTube songs and group them into playlists by genre similarity. This was a fun idea I had and spent about a week on, so it definitely has room for improvement. That being said, I'm happy with it!

### How to run
A brief overview of the intended use:
 - Use `playlist_data.py` to download the target playlist to cluster
 - Optional: finetune CLAP model with `finetune_CLAP.py` on downloaded .mp4 songs for better embeddings
 - Get embeddings from `CLAP_model.py` by setting the CLAP model and playlist directory
 - Generate clusters with `cluster_embeddings.py`, using K-Means (recommended) or HDBSCAN
 - Visualize cluster performance with `visualize_clusters.py`
 - Upload the generated clusters back to YouTube with `upload_cluster.py`

### Notes
This project is at a functional point, but still leaves some things to be desired. Some things to note:   
 - K-means has been the most successful clustering model. I suggest to do some analysis on your playlist to see how many genres there are, and try that many K clusters. Even if the clustering metric for that K look bad, ultimately the only real metric that matters is how they sound. Give the split a try, and then experiment with other options. Clusters are subjective!
 - HDBSCAN seems to always have a lot of songs in the "noise" category regardless of the hyperparameter selection. If your playlist has a lot of noise then this is a good choice, but otherwise it's rather unhelpful.
 - The remaining clustering models were much worse and are simply there for the No Free Lunch theorem.
 - The using the YouTube API in `upload_cluster.py` only works for playlists with ~150 songs or less, as you otherwise run out of individual credits. Manual upload isn't that bad with the Multiselect YouTube Extension: `https://chromewebstore.google.com/detail/gpgbiinpmelaihndlegbgfkmnpofgfei?utm_source=item-share-cb`

Future aims:
 - Keep trying different splits on smaller dataset with K-means to find best number of clusters!
 - Add an LLM into clusterization with title, author, and description information for more accuracy with the bigger dataset 