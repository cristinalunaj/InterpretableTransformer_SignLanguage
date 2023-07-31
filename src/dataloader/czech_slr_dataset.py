import ast
import torch

import pandas as pd
import numpy as np
import torch.utils.data as torch_data


BODY_IDENTIFIERS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist"
]
HAND_IDENTIFIERS = [
    "wrist",
    "indexTip",
    "indexDIP",
    "indexPIP",
    "indexMCP",
    "middleTip",
    "middleDIP",
    "middlePIP",
    "middleMCP",
    "ringTip",
    "ringDIP",
    "ringPIP",
    "ringMCP",
    "littleTip",
    "littleDIP",
    "littlePIP",
    "littleMCP",
    "thumbTip",
    "thumbIP",
    "thumbMP",
    "thumbCMC"
]
HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]


def load_dataset(file_location: str):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    # TEMP
    labels = df["labels"].to_list()
    # labels = [label + 1 for label in df["labels"].to_list()]
    data = []

    for row_index, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, labels


def load_dataset_mediaPipe(file_location: str, n_landm=42):
    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    # TEMP
    labels = []
    data = []
    for row_index, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row[df.columns[0]])), n_landm, 2))
        labels.append(ast.literal_eval(row["labels"])[0])
        for land_i in range(n_landm):
            current_row[:, land_i, 0] = ast.literal_eval(row[df.columns[0::2][land_i]]) #X
            current_row[:, land_i, 1] = ast.literal_eval(row[df.columns[1::2][land_i]]) #Y
        data.append(current_row)
    return data, labels



def tensor_to_dictionary(landmarks_tensor: torch.Tensor, n_landm=54) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index in range(n_landm):
        output[landmark_index] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict, n_landm=54) -> torch.Tensor:

    output = np.empty(shape=(len(landmarks_dict[0]), n_landm, 2))

    for landmark_index in range(n_landm):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[landmark_index]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[landmark_index]]

    return torch.from_numpy(output)



class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, num_labels=5, transform=None,mediapipe=False, n_landmarks=54):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        if(mediapipe):
            loaded_data = load_dataset_mediaPipe(dataset_filename, n_landm=n_landmarks)
        else:
            loaded_data = load_dataset(dataset_filename)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = num_labels
        self.transform = transform
        self.n_landmarks = n_landmarks



    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx]]) # - 1

        #depth_map = tensor_to_dictionary(depth_map, n_landm=self.n_landmarks)
        #depth_map = dictionary_to_tensor(depth_map, n_landm=self.n_landmarks)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)



class WLASLnewDataset(CzechSLRDataset):
    def __init__(self, df_distribution, splitSet, dataset_filename: str, num_labels=5, transform=None):
        super().__init__(dataset_filename, num_labels, transform)
        self.df_distribution = df_distribution
        self.splitSet = splitSet




