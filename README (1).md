# Facial Emotion Recognition with Attendance System

A deep learning project that detects human emotions from facial images in real time, built on a custom CNN trained on the FER-2013-style dataset. It goes beyond single-image classification into a live webcam pipeline that performs face detection, face recognition, emotion analysis, and automatic attendance logging.

## Features

- **CNN-based emotion classifier** trained from scratch on 48×48 grayscale facial images across 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- **Single-image prediction** — upload a photo and get the predicted emotion
- **Face detection** using OpenCV Haar Cascades, with bounding boxes and emotion labels drawn on the image
- **Live webcam attendance system** — combines `face_recognition` for identity matching and `DeepFace` for real-time emotion analysis, logging each recognized person's name, date, time, and detected emotion to a CSV file

## Project Structure

```
.
├── FacialEmotionRecognition.ipynb   # Main notebook: training + full pipeline
├── attendance.csv                    # Generated attendance log (created at runtime)
├── emotiondetector.h5                # Trained model (generated after training)
└── README.md
```

## Model Architecture

A sequential CNN with three convolutional blocks followed by fully connected layers:

- **Block 1–3:** Conv2D (64 → 128 → 256 filters) + BatchNormalization + Conv2D + MaxPooling2D + Dropout
- **Dense layers:** 256 → 128 units with BatchNormalization/Dropout
- **Output:** 7-unit softmax layer (one per emotion class)

Trained with the Adam optimizer and categorical cross-entropy loss.

## Dataset

The model expects a dataset with `train/` and `validation/` folders, each containing one subfolder per emotion label (e.g. `train/happy/`, `train/sad/`), such as the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). Place it as `archive.zip` in your Google Drive (if using Colab) or extract it locally and update the paths in the notebook.

## Setup

### Option A — Google Colab (training + single-image demo)

1. Upload your dataset as `archive.zip` to Google Drive (`MyDrive/archive.zip`)
2. Open the notebook in Colab and run the cells top to bottom through the training section
3. The trained model saves to `MyDrive/emotiondetector.h5`

### Option B — Local environment (live webcam attendance system)

The webcam attendance section (face recognition + live emotion detection) requires direct camera and display access, which Colab does not provide. Run this part locally instead:

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

Then run the notebook locally (Jupyter/JupyterLab) starting from the attendance-system cells, using your already-trained `emotiondetector.h5` model.

## Requirements

```
tensorflow
numpy
pandas
opencv-python
matplotlib
scikit-learn
tqdm
face_recognition
deepface
```

> `face_recognition` depends on `dlib`, which may require CMake and a C++ compiler to install. On Windows, installing via `conda` is often easier than `pip`.

## Usage

1. **Train the model** (or use a pre-trained `emotiondetector.h5`)
2. **Test on a single image** — upload a photo and view the predicted emotion
3. **Run live attendance** — upload a reference photo of each known person, then start the webcam loop:
   - Detects and recognizes faces in the video feed
   - Predicts each recognized person's current emotion via DeepFace
   - Logs one attendance entry per person per session to `attendance.csv`
   - Press `q` to stop the webcam loop

## Results

Training/validation accuracy and loss curves are plotted at the end of training. Update this section with your own final test accuracy once you've trained the model, e.g.:

```
Test Accuracy: XX.XX%
```

## Known Limitations

- Emotion recognition accuracy is inherently limited by dataset quality and class imbalance (e.g. `disgust` is typically underrepresented in FER-2013)
- Face recognition currently supports one known identity at a time per enrollment step; extend `known_encoding`/`known_name` to lists for multi-person recognition
- The Colab-based demo cells (browser photo capture, file upload) and the local webcam attendance cells are separate workflows and cannot both run in the same environment


## Acknowledgments

- [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [OpenCV](https://opencv.org/) for Haar Cascade face detection
- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [DeepFace](https://github.com/serengil/deepface) for emotion analysis
