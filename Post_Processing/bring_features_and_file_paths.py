import os
import torch
import numpy as np
import datetime

np.random.seed(seed=10)


# Brings the feature vectors from a file if features_path is a file path
# If features_path is a folder, iterates thorught one level internal folders
# and brings feat.pth from each one.
# Then cats the tensors together.
def bring_features_and_file_paths(path, sub_sample=None):
    if os.path.isdir(path):
        firstFolder = True
        for folder in os.listdir(path):
            if firstFolder:
                auxiliary = torch.load(os.path.join(path, folder, "feat.pth"))
                feats = auxiliary
                auxiliary = torch.load(os.path.join(path, folder, "file_path.pth"))
                file_paths = auxiliary
                firstFolder = False
            else:
                auxiliary = torch.load(os.path.join(path, folder, "feat.pth"))
                feats = torch.cat((feats, auxiliary), 0)
                auxiliary = torch.load(os.path.join(path, folder, "file_path.pth"))
                if auxiliary.shape[1] > file_paths.shape[1]:
                    auxiliary = auxiliary[:, : file_paths.shape[1]]
                elif auxiliary.shape[1] < file_paths.shape[1]:
                    file_paths = file_paths[:, : auxiliary.shape[1]]

                file_paths = torch.cat((file_paths, auxiliary), 0)
    else:
        print("You need to specify a directory")
        return 1

    assert feats.shape[0] == file_paths.shape[0]
    if sub_sample:
        assert sub_sample > 0.0 and sub_sample < 1.0
        number_of_rows = feats.shape[0]
        n_samples = int(feats.shape[0] * sub_sample)
        random_indices = np.random.choice(number_of_rows, size=n_samples, replace=False)
        feats = feats[random_indices, :]
        file_paths = file_paths[random_indices, :]
        print("We have {} feature vectors.".format(feats.shape[0]))
        return feats, file_paths, random_indices

    print("We have {} feature vectors.".format(feats.shape[0]))
    return feats, file_paths


def transform_features(feats, scale, dim_red):
    return dim_red.transform(scale.transform(feats))


def from_spectrogram_path_to_BirdNET_output_path(spectrogram_path):
    file_path = "".join([chr(int(x)) for x in spectrogram_path]).replace("~", "")
    path, file = os.path.split(file_path)
    return os.path.join(path, file[:24] + ".BirdNET.selections.txt")


def get_spectrogram_time_mark_in_file(spectrogram_path, spectrogram_duration):
    file_path = "".join([chr(int(x)) for x in spectrogram_path]).replace("~", "")
    path, file = os.path.split(file_path)
    file, _ = os.path.splitext(file)
    return int(file[25:]) * spectrogram_duration


def get_BirdNET_detections(fpath, spectrogram_interval, confidence_threshold=0.0):
    dpath, auxiliary = os.path.split(fpath)

    year = int(auxiliary[:4])
    month = int(auxiliary[4:6])
    day = int(auxiliary[6:8])
    hour = int(auxiliary[9:11])
    minute = int(auxiliary[11:13])
    second = int(auxiliary[13:15])

    updated_date = datetime.datetime(year, month, day, hour, minute, second)
    updated_date = updated_date + datetime.timedelta(seconds=spectrogram_interval[0])
    year = updated_date.year
    month = updated_date.month
    day = updated_date.day
    hour = updated_date.hour
    minute = updated_date.minute
    second = updated_date.second

    week = datetime.date(year, month, day).isocalendar()[1]
    weekday = datetime.date(year, month, day).isocalendar()[2]

    device = dpath.split("/")[-2]
    SET = dpath.split("/")[-3]

    try:
        with open(fpath) as f:
            lines = [line.rstrip() for line in f]
    except IOError:
        data = []
        detection = "No detection"
        confidence = 1.0
        data.append(
            {
                "detection": detection,
                "confidence": confidence,
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
                "minute": minute,
                "second": second,
                "week": week,
                "weekday": weekday,
                "device": device,
                "set": SET,
            }
        )
        return data

    data = []
    for line in lines[1:]:
        start = float(line.split()[5])
        end = float(line.split()[6])
        if get_overlapping(spectrogram_interval, (start, end)) > 0.7:
            detection = " ".join(line.split()[10:-2])
            confidence = float(line.split()[-2])
            if confidence > confidence_threshold:
                data.append(
                    {
                        "detection": detection,
                        "confidence": confidence,
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                        "second": second,
                        "week": week,
                        "weekday": weekday,
                        "device": device,
                        "set": SET,
                    }
                )

    if len(data) == 0:
        detection = "No detection"
        confidence = 1.0
        data.append(
            {
                "detection": detection,
                "confidence": confidence,
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
                "minute": minute,
                "second": second,
                "week": week,
                "weekday": weekday,
                "device": device,
                "set": SET,
            }
        )
        
    return data


def get_overlapping(interval1, interval2):
    start1 = interval1[0]
    end1 = interval1[1]
    start2 = interval2[0]
    end2 = interval2[1]

    assert start1 < end1 and start2 < end2

    length1 = end1 - start1
    length2 = end2 - start2

    left = max(start1, start2)
    right = min(end1, end2)

    return (right - left) / min(length1, length2)
