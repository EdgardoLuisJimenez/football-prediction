
from scipy import cluster
from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape the image into 2d Array
        image_2d = image.reshape(-1, 3)

        # Perform k-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        
        return kmeans


    def get_player_color(self, frame, bbox) -> np.ndarray:
        # frame [height/rows, width/cols]
        # bbox is [x1, y1, x2, y2] (left, top, right, bottom)
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # image[vertical_range, horizontal_range]
        top_half_image = image[0: int(image.shape[0]/2), :]

        # Get the clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel 
        # kmeans.labels_ = [0, 0, 1, 1, 0, 1]
        # pixel 0 → cluster 0
        # pixel 1 → cluster 0
        # pixel 2 → cluster 1
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [
                            clustered_image[0,0], # Top-left corner
                            clustered_image[0,-1], # Top-right corner
                            clustered_image[-1, 0], # Bottom-left corner
                            clustered_image[-1,-1] # Bottom-right corner
                            ]

        # Return the one that appears most often in corner_clusters
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Pick only the player's clusters and store the 
        # RGB value in player_color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        # All the RGB value of the all players
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id
        
        return team_id
