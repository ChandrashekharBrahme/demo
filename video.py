# input video --> read_video  --> face_detection (MP)  ---> resize  ---> chunking  --->
import torch
import os
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from TS_CAN import TSCAN
from tqdm import tqdm
from evaluation.metrics import calculate_metrics
from statistics import mean
# def read_video(video_file):
#     """Reads a video file, returns frames(T,H,W,3) """
#     VidObj = cv2.VideoCapture(video_file)
#     VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)  # to set the value of CAP_PROP_POS_MSEC to 0
#     success, frame = VidObj.read()
#     frames = list()
#     while success:
#         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
#         frame = np.asarray(frame)
#         frames.append(frame)
#         success, frame = VidObj.read()
#
#     print("\n no of frames in read_video function \n:", len(frames))
#     return np.asarray(frames)


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
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.double)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.double)
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

def process(processed_frames):
    data=list()
    f_c = processed_frames.copy()
    print("processed frames shape ", np.array(f_c).shape)
    data.append(diff_normalize_data(f_c))
    data.append(standardized_data(f_c))

    #print("\n before concat \n", data)
    print("\nshape before\n", np.array(data).shape)
    data = np.concatenate(data, axis=-1)
    #print("\n after concat \n", data)
    print("\n shape after\n", np.array(data).shape)
    frames_clips = chunk(data)
    return frames_clips


def prediction(data_loader, model_path='A:/Pycharm/newfinaldemo/PURE_TSCAN.pth', device='cpu'):

    predictions = dict()
    chunk_len=210
    #  self.base_len = self.num_of_gpu * self.frame_depth
    base_len = 1* 10

    model = TSCAN(frame_depth=10, img_size=72).to(device)
    model = torch.nn.DataParallel(model, device_ids=list(range(1)))

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.double()
    model = model.to(device)
    model.eval()



    with torch.no_grad():
        for data_dict in tqdm(data_loader, ncols=80):
            video_name = data_dict['video_name']
            batch_number = data_dict['batch_number']
            frames = data_dict['frames']
            batch_size = frames.shape[0]
            print("\nbatch size:", batch_size)

            data_test = frames.to(device, dtype=torch.double)
            N, D, C, H, W = data_test.shape
            print(f' N{N}, D{D}, H{H}, W{W} ,C{C} ')
            data_test = data_test.view(N * D, C, H, W)
            data_test = data_test[:(N * D) // base_len * base_len]

            pred_ppg_test = model(data_test)

            for idx in range(batch_size):
                subj_index = video_name
                print(f"subj_index",subj_index)
                sort_index = batch_number[idx]
                print(f"sort_index{idx}", sort_index)
                if subj_index not in predictions.keys():
                    predictions[subj_index] = dict()
                predictions[subj_index][sort_index] = pred_ppg_test[idx * chunk_len:(idx + 1) * chunk_len]

    print('\n this is predictions \n', len(predictions))


    hr_rate = calculate_metrics(predictions)
    print("hr mean:", hr_rate)

    return hr_rate

def frame_calculation(frames_array):
    frames = frames_array
    print(" in video.py:",frames)
    face_processing = FacePreprocessing()
    processed_frames = face_processing.process_frames(frames)
    print("\nprocessed frames after MP\n", list(processed_frames.shape))
    # frames after processing --> retun noramlize , chunk
    final_frames = process(processed_frames)

    # required as tesnor and  to fit in NDCHW data type mode

    final_data = np.float32(final_frames)

    transposed_dim = np.transpose(final_data, (0, 1, 4, 2, 3))
    print('after transpose ', transposed_dim.shape)
    test_data = torch.tensor(transposed_dim)

    data_dict = {
        'frames': torch.tensor(transposed_dim),
        'video_name': 's23_T1',
        'batch_number': [0, 1, 2, 3]  # Assuming there's only one batch
    }
    data_loader = [data_dict]

    predict_data = prediction(data_loader)
    print(predict_data)
    return predict_data


if __name__ == "__main__":
    pass
