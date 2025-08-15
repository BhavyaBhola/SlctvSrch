-----
# SlctvSrch: Video Surveillance and Object Extraction Tool

SlctvSrch is a powerful Python tool designed for **video surveillance**, **background extraction**, and **selective object tracking**. It excels at analyzing video footage from a stationary camera, allowing you to extract a stable background and isolate the movements of specific objects based on their unique IDs. This is ideal for applications like security monitoring, traffic analysis, and motion-based event detection.

-----

## Features

  - **Background Subtraction:** Automatically identifies and extracts the static background from a video, even while objects are in motion. This provides a clean reference frame for further analysis.
  - **Object Tracking:** Tracks and assigns a unique, persistent **track ID** to each moving object detected in the video.
  - **Selective Track Retrieval:** Isolate and retrieve all frames for a specific object using its track ID. This feature is perfect for focusing on a single point of interest.
  - **Sample Data Included:** The repository comes with sample `.mp4` videos and `.jpg` images, making it easy to test the tool immediately.

-----

## How It Works

SlctvSrch utilizes a sophisticated pipeline of computer vision techniques to achieve its functionality:

1.  **Object Detection and Masking:** The tool first employs a combination of **YOLO (You Only Look Once)** for object detection and **SAM (Segment Anything Model)** to generate precise masks for each object. This ensures accurate identification and isolation of moving subjects.
2.  **Background Extraction:** The static background is extracted using a **temporal median** filter applied over a predefined number of video frames. This method effectively "removes" moving objects, leaving a clean, static image of the scene.
3.  **Robust Tracking:** **Kalman filters** are used to predict the position of each tracked object. This is a crucial step that allows the system to maintain a persistent track ID for objects even during **occlusions** (when they are temporarily blocked from view), significantly improving tracking reliability.
4.  **Object Stitching:** Once an object's mask is extracted, it is seamlessly stitched back onto the clean background using **alpha blending**, creating a final output that isolates the object's movement on a static backdrop.

-----

## Prerequisites and Installation

To get started, you'll need **Python 3.x** and the required libraries. The tool is built using common computer vision libraries.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BhavyaBhola/SlctvSrch.git
    cd SlctvSrch
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

-----

## Usage

⚠️ **Important:** This tool is designed for videos captured with a **stationary camera**. Any significant camera movement will affect the accuracy of background extraction and object tracking.

### Step 1: Start Tracking and Extract Background

This command processes a video, performs object tracking, and generates two key outputs:

1.  A new video showing the tracked objects with bounding boxes and their assigned IDs.
2.  An image file containing the extracted static background.

<!-- end list -->

```bash
python startTracking.py --video_path <path/to/your/video.mp4>
```

**Example:**
To process the included sample video, run:

```bash
python startTracking.py --video_path sample_video.mp4
```

### Step 2: Extract a Specific Track

After running `startTracking.py`, you can use the track IDs from the output video to extract frames for a specific object. This command will create a new directory containing all frames where the specified object was detected.

```bash
python extTrack.py --track_id <track_id>
```

**Example:**
If the output video shows an object with `ID: 3` that you want to isolate, run:

```bash
python extTrack.py --track_id 3
```

-----

## Example Outputs

### 🎥 Input and Tracking Video

  - **Sample Input Video:** [▶ Watch Sample Video](https://github.com/user-attachments/assets/311b6adf-0941-4e05-8ba2-61cdc985024b)
  - **Tracking Video Output:** The output video will have bounding boxes and IDs assigned to each moving object.
  - [![Watch the video](https://github.com/BhavyaBhola/SlctvSrch/blob/main/background.jpg)](https://github.com/user-attachments/assets/49ce013b-471c-41c7-919a-1753b53755e5)

### 📸 Track Extraction Frames

After running `extTrack.py`, a folder will be created with individual frames of the selected object. For instance, extracting **Track ID 7** would produce a series of images like these:
<div align="center">
  <img src="https://github.com/user-attachments/assets/e0ffcf51-7758-4873-894c-81fb32c990b7" width="80%" />
  <img src="https://github.com/user-attachments/assets/f919668d-28c0-49c3-a493-6f2936e85434" width="80%" />
  <img src="https://github.com/user-attachments/assets/e52b11fb-56ef-4f9a-b32d-196e348ceeab" width="80%" />
</div>

-----

## Contributing

We welcome contributions\! If you have ideas for new features, bug fixes, or improvements, please feel free to:

  - **Open an issue** to report bugs or suggest new features.
  - **Create a pull request** with your code changes.
