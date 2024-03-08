# input video --> read_video  --> face_detection (MP)  ---> resize  ---> chunking  --->
import os
import numpy as np
import cv2
import mediapipe as mp

def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)  # to set the value of CAP_PROP_POS_MSEC to 0
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()

    print("\n no of frames in read_video function \n:", len(frames))
    return np.asarray(frames)



class FacePreprocessing:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection()
        self.face_size = (72, 72)

    def process_frames(self, frame_list):
        processed_frames = []
        for frame in frame_list:
            # Use mediapipe for face detection
            result = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.detections:
                # Get the bounding box of the face
                bboxC = result.detections[0].location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop the face
                face = frame[y:y + h, x:x + w]

                # Resize the face frames
                resized_face = cv2.resize(face, self.face_size, interpolation=cv2.INTER_AREA)

                # Append the processed face to the list
                processed_frames.append(resized_face)

        return np.asarray(processed_frames)

def chunk(frames, chunk_length):

    clip_num = frames.shape[0] // chunk_length
    frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    print("\n frame_clips in chunk function: \n",len(frames_clips))
    return np.array(frames_clips)

if __name__ == "__main__":
    input_video = "A:/final_project/test/s23/vid_s23_T2.mp4"
    input_dir = os.path.join(input_video)
    frames= read_video(input_dir)
    #print(len(frames))

    face_processing = FacePreprocessing()
    processed_frames = face_processing.process_frames(frames)
    print(len(processed_frames))

    chunks = chunk(processed_frames, 210)
    print("\n these are chunks\n", list(chunks.shape))

    pass


