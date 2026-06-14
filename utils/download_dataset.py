import kagglehub

# Download latest version
path = kagglehub.dataset_download("antonkozyriev/game-recommendations-on-steam", output_dir="../dataset/")

print("Path to dataset files:", path)