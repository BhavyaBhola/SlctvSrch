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
### !!! Make sure there is no change in the background of video or video should be made using a still camera
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

[▶ Watch Sample Video](https://github.com/user-attachments/assets/311b6adf-0941-4e05-8ba2-61cdc985024b)

### 📸 Sample Tracking video

[![Watch the video](https://github.com/BhavyaBhola/SlctvSrch/blob/main/background.jpg)](https://github.com/user-attachments/assets/49ce013b-471c-41c7-919a-1753b53755e5)

### 📸 Sample Track extraction frames

The output here is shown for track id (7) from the above tracking video.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e0ffcf51-7758-4873-894c-81fb32c990b7" width="30%" />
  <img src="https://github.com/user-attachments/assets/f919668d-28c0-49c3-a493-6f2936e85434" width="30%" />
  <img src="https://github.com/user-attachments/assets/e52b11fb-56ef-4f9a-b32d-196e348ceeab" width="30%" />
</div>




---

## Contributing

Contributions and feature requests are welcome! Feel free to raise issues or create pull requests.

---

