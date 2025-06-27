import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/rsna-bone-age")

print("Path to dataset files:", path)