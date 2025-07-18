# Demo the result:
https://github.com/user-attachments/assets/7854435d-d2ee-41d1-9d1f-df63a80b5c1a

# Tracking and Detecting the Winner in a Running Race 🏁

I trained a computer vision model using datasets collected from Roboflow to detect and track athletes in 100 meter videos. The system leverages YOLOv8 for fast and accurate runner detection, and ByteTrack to assign and follow unique IDs for each athlete throughout the race.

The application automatically detects runners, tracks each player with a unique ID, draws a virtual finish line, and identifies the winner based on who crosses the finish line first in the video.

Training the model for 50 epochs took approximately 6 hours on my M4 Max.
## Features

- **YOLOv8 Detection:** Fast and accurate person detection on race video frames.
- **ByteTrack Tracking:** Assigns a unique ID to each runner and tracks across frames.
- **Finish Line Logic:** Visually shows the finish line; only recognizes the winner after the line appears.
- **Winner Detection:** Prints out which player (by track ID) first crosses the finish line, excluding late joiners/coaches.
- **Easy Video Output:** Saves annotated video with bounding boxes, IDs, and finish line overlays.

## Dataset

The model is trained on a custom dataset of ~4,000 annotated images, **combined from two public Roboflow sources**:

- [Running Athlete Dataset](https://universe.roboflow.com/gg-3i5yz/running-cefql) (`running-cefql`)
- [KMUTT-AI Running Dataset](https://universe.roboflow.com/kmutt-ai/running-muumz) (`running-muumz`)
- [Human Running t6vec](https://universe.roboflow.com/anton-mre5k/human-running-t6vec-w4srv) this dataset cannot download, so I skiped

All images are labeled for a single `person` class to focus on accurate athlete tracking and race analysis.
