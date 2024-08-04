from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # This method should return a KMeans model trained on the image data.
        if image.size == 0:
            return None
        # Flatten the image and apply KMeans clustering (example parameters)
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Ensure bbox values are integers
        x, y, w, h = bbox
        x, y, w, h = map(int, [x, y, w, h])

        # Debugging print statements
        print(f"Bounding box values - x: {x}, y: {y}, w: {w}, h: {h}")

        # Check frame boundaries to avoid index errors
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        # Debugging print statements
        print(f"Adjusted bounding box values - x: {x}, y: {y}, w: {w}, h: {h}")

        # Crop the player image from the frame
        player_image = frame[y:y+h, x:x+w]
        if player_image.size == 0:
            return None  # Return None if the cropped image is empty

        # Extract the top half of the player image
        top_half_image = player_image[:h//2, :, :]

        # Get the KMeans model for clustering
        kmeans = self.get_clustering_model(top_half_image)
        if kmeans is None:
            return None  # Return None if clustering failed

        # Return the color of the first cluster center
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
