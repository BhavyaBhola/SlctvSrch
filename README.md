````markdown
# SlctvSrch

A tool for surveillance, tracking, and object extraction from videos—ideal for background extraction and selective track retrieval.

---

## Features
- Extract background frames while tracking objects in a video.
- Retrieve and isolate specific object tracks using their track IDs.
- Comes with sample `.jpg` images and `.mp4` videos for quick testing.

---

## Usage

### 1. Start Tracking & Extract Background  
```bash
python startTracking.py --video_path <video_path>
````

### 2. Extract a Specific Track

```bash
python extTrack.py --track_id <track_id>
```

#### Example

```bash
python startTracking.py --video_path sample_video.mp4
python extTrack.py --track_id 3
```

---

## Example Output

### 🎥 Sample Input Video

[▶ Watch Sample Video](sample_video.mp4)

### 📸 Sample Tracking video

<video src="https://github.com/BhavyaBhola/SlctvSrch/blob/main/output_video.mp4"></video>

### 📸 Sample Track extraction video

<video src="https://github.com/BhavyaBhola/SlctvSrch/blob/main/extTrack.mp4c"></video>


---

## Contributing

Contributions and feature requests are welcome! Feel free to raise issues or create pull requests.

---

