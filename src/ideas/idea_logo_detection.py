#logo detection and handling

# IDEAS:
# 1. Use a pre-trained model to detect logos in images.
# 2. Crop or mask the detected logos before processing the image.

# Working process:
# 1. Load the images.
# 2. Use similarity search between them. (Planets are a good example why it"s not the best idea) OR very high similarity needed to flag as duplicate.
# 3. If similarity is high keep only one and then put the same description for all of them.
# 4. If similarity is low, process normally for logo detection.
    # 4.1. Use a pre-trained model to detect logos in images.   --- IS THIS REALLY NEEDED?
