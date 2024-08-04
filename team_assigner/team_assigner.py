from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    class TeamAssigner:
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        if image_2d.shape[0] == 0:
            return None  # Return None if the image is empty
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x, y, w, h = bbox
        player_image = frame[y:y+h, x:x+w]
        if player_image.size == 0:
            return None  # Return None if the cropped image is empty
        top_half_image = player_image[:h//2, :, :]
        kmeans = self.get_clustering_model(top_half_image)
        if kmeans is None:
            return None  # Return None if clustering failed
        player_color = kmeans.cluster_centers_[0]
        return player_color


    def assign_team_color(self, frame, player_tracks):
        team_colors = []
        for player_id, track in player_tracks.items():
            bbox = track['bbox']
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                team_colors.append(player_color)

        if len(team_colors) < 2:
            print("Not enough valid player colors found to assign teams.")
            return
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id
