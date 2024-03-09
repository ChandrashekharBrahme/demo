# input video --> read_video  --> face_detection (MP)  ---> resize  ---> chunking  --->
import os
import numpy as np
import cv2
import mediapipe as mp
import torch
import TSCAN
import tqdm

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

def chunk(frames, chunk_length=210):

    clip_num = frames.shape[0] // chunk_length
    frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    print("\n frame_clips in chunk function: \n",len(frames_clips))
    return np.array(frames_clips)


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    print(f"\n N:{n}  C;{c} H:{h} W:{w} \n")
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data

def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data

def process(frames):
    data=list()
    f_c = processed_frames.copy()
    data.append(diff_normalize_data(f_c))
    data.append(standardized_data(f_c))

    #print("\n before concat \n", data)
    print("\nshape before\n", np.array(data).shape)
    data = np.concatenate(data, axis=-1)
    #print("\n after concat \n", data)
    print("\n shape after\n", np.array(data).shape)
    frames_clips = chunk(data)
    return frames_clips

def prediction(self):
    data_loader=[]
    data_loader=self.process(frames)

    predictions = dict()
    self.model = TSCAN(frame_depth=self.frame_depth, img_size=72).to(self.device)
    self.model = torch.nn.DataParallel(self.model, device_ids=list(range(1)))
    path='C:/Users/CHANDRASHEKHAR/Desktop/latest/PURE_TSCAN.pth'

    self.model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    print("Testing uses pretrained model!")

    self.model = self.model.to(self.config.DEVICE)
    self.model.eval()

    print("Running model evaluation on the testing dataset!")

    with torch.no_grad():
        for _, test_batch in enumerate(tqdm(data_loader, ncols=80)):
            batch_size = data_loader[0].shape[0]
            data_test = test_batch[0].to(
                    self.config.DEVICE)
            N, D, C, H, W = data_test.shape
            data_test = data_test.view(N * D, C, H, W)

            data_test = data_test[:(N * D) // self.base_len * self.base_len]
            pred_ppg_test = self.model(data_test)

            if self.config.TEST.OUTPUT_SAVE_DIR:
                # labels_test = labels_test.cpu()
                pred_ppg_test = pred_ppg_test.cpu()

            print("\n batch size: \n ", batch_size)

            for idx in range(batch_size):

                subj_index = test_batch[2][idx]
                print("\n subj_index: \n ", subj_index)

                sort_index = int(test_batch[3][idx])
                print("\n this is sort_index:  \n",sort_index)

                if subj_index not in predictions.keys():
                    predictions[subj_index] = dict()        # making dict of predictions
                    #labels[subj_index] = dict()
                predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                #labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

    print('')
    calculate_metrics(predictions,self.config)

    pass





if __name__ == "__main__":
    input_video = "A:/final_project/test/s23/vid_s23_T2.mp4"
    input_dir = os.path.join(input_video)
    frames= read_video(input_dir)
    #print(len(frames))

    face_processing = FacePreprocessing()
    processed_frames = face_processing.process_frames(frames)

    print("\nprocessed frames after MP\n",list(processed_frames.shape))

    final_frames = process(processed_frames)
    #print("\nfinal frames\n", final_frames)
    print("\n shape of final\n", final_frames.shape)
    #print("\nchunks:\n", chunks)
    #print("\n these are chunks\n", list(chunks.shape))

    pass


