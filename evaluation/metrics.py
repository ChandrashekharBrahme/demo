import torch
import numpy as np
import pandas as pd

from evaluation.post_process import *
from tqdm import tqdm


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()

    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])

        video_frame_size = prediction.shape[0]

        print(f'\nvdo frm size:{video_frame_size}')
        # if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        if True:
            window_frame_size = 2 * 35
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i + window_frame_size]
            # label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(
                    f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            pred_hr_fft = calculate_metric_per_video(pred_window, diff_flag=True, fs=35, hr_method='FFT')
            predict_hr_fft_all.append(pred_hr_fft)

    print("\n mean:", mean(predict_hr_fft_all))

