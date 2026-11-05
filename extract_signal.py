# extract_signal.py
# What this script does:
#   1. Opens the video  data/me.mp4
#   2. Finds your face in every frame using MediaPipe Face Mesh
#   3. Cuts out the forehead area
#   4. Averages the Red, Green and Blue colour values of that area for each frame
#   5. Plots the Green values over time (in seconds)

# --- Imports: these load extra tools (libraries) that we need ---
import cv2                        # OpenCV: reads video files and handles images
import mediapipe as mp            # MediaPipe: finds the face and its landmark points
import numpy as np                # NumPy: fast maths on lists/arrays of numbers
import matplotlib.pyplot as plt   # Matplotlib: draws the graph

# --- Settings you can change ---
VIDEO_PATH = "data/me.mp4"        # where the video file is (relative to the repo folder)

# Face Mesh gives 468 numbered points on the face.
# These numbers are the points that outline the forehead (just above the eyebrows).
# Joining them in this order draws a shape around the forehead.
FOREHEAD_POINTS = [67, 109, 10, 338, 297, 336, 9, 107]

# --- Open the video ---
video = cv2.VideoCapture(VIDEO_PATH)          # open the video file
if not video.isOpened():                       # if it could not be opened...
    raise SystemExit(f"Could not open {VIDEO_PATH}. Is the path right?")  # ...stop with a message

fps = video.get(cv2.CAP_PROP_FPS)              # frames per second of the video
if fps <= 0:                                   # some files do not report fps
    fps = 30.0                                 # so we guess 30 as a fallback
print(f"Video speed: {fps:.2f} frames per second")

# --- Set up the face finder ---
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,       # False = treat input as a video (tracks the face between frames, faster)
    max_num_faces=1,               # we only expect one face (you)
    refine_landmarks=True,         # slightly more accurate points around eyes and lips
    min_detection_confidence=0.5,  # how sure MediaPipe must be to say "this is a face"
    min_tracking_confidence=0.5,   # how sure it must be to keep following the face
)

# --- Empty lists where we store one number per frame ---
reds, greens, blues = [], [], []   # average colour values for each frame
frame_count = 0                    # counts how many frames we have processed

# --- Main loop: go through the video one frame at a time ---
while True:
    ok, frame = video.read()       # read the next frame; ok is False when the video ends
    if not ok:                     # no more frames
        break                      # leave the loop

    frame_count += 1               # one more frame processed
    height, width = frame.shape[:2]  # the frame's size in pixels (height first, then width)

    # OpenCV loads colours in BGR order, but MediaPipe wants RGB, so we swap them
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)   # ask MediaPipe to find the face in this frame

    # If no face was found, save "NaN" (not a number) so the time axis stays correct
    if not results.multi_face_landmarks:
        reds.append(np.nan)
        greens.append(np.nan)
        blues.append(np.nan)
        continue                   # skip to the next frame

    landmarks = results.multi_face_landmarks[0].landmark  # the points of the first (only) face

    # Landmark positions are given from 0 to 1, so we multiply by the image size to get pixels
    points = np.array(
        [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in FOREHEAD_POINTS],
        dtype=np.int32,
    )

    # Make a black image (mask) the same size as the frame, one channel
    mask = np.zeros((height, width), dtype=np.uint8)
    # Fill the forehead shape with white (255). White = "use these pixels"
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

    # Average the colour of only the white (forehead) pixels. Returns (R, G, B, unused)
    mean_r, mean_g, mean_b, _ = cv2.mean(rgb, mask=mask)

    reds.append(mean_r)            # save the average red for this frame
    greens.append(mean_g)          # save the average green for this frame
    blues.append(mean_b)           # save the average blue for this frame

# --- Clean up ---
video.release()                    # close the video file
face_mesh.close()                  # free the face finder
print(f"Processed {frame_count} frames")

# --- Turn the lists into NumPy arrays so they are easier to work with ---
reds = np.array(reds)
greens = np.array(greens)
blues = np.array(blues)

# Time in seconds for each frame = frame number divided by frames per second
time_seconds = np.arange(frame_count) / fps

# Tell the user if the face was missed in some frames
missing = int(np.isnan(greens).sum())   # how many frames had no face
if missing:
    print(f"Warning: no face found in {missing} of {frame_count} frames")

# --- Plot the green values ---
plt.figure(figsize=(10, 4))                    # create a graph window, 10 inches wide, 4 tall
plt.plot(time_seconds, greens, color="green")  # draw green values (y) against time (x)
plt.xlabel("Time (seconds)")                   # label for the bottom axis
plt.ylabel("Average green value (0-255)")      # label for the side axis
plt.title("Forehead green channel over time")  # title on top
plt.grid(True)                                 # light grid lines to make it easier to read
plt.tight_layout()                             # stop labels being cut off
plt.savefig("green_signal.png", dpi=150)       # also save the graph as a picture file
plt.show()                                     # open the graph window
