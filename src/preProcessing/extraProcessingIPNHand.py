import os.path

import pandas as pd
import numpy as np
import argparse

def compact_data_IPNHand(videos_path, path_DS_distribution, poses_path, out_dir):
    videos_list = pd.read_csv(videos_path, header=None)
    df_annotations = pd.read_csv(path_DS_distribution)

    # Load the first
    current_video = videos_list.loc[0][0]
    df = pd.read_csv(os.path.join(poses_path,current_video + "_poses_landamarks.csv"))
    df['label'] = np.zeros(len(df))
    df_annotations_video = df_annotations[df_annotations["video"] == current_video]
    df_annotations_video = df_annotations_video.reset_index()

    for i in range(len(df_annotations_video)):
        df.loc[(df_annotations_video["t_start"][i] <= df["frame"]) & (
                    df["frame"] <= df_annotations_video["t_end"][i]), "label"] = df_annotations_video["id"][i]

    # Load the rest
    for j in range(1, len(videos_list)):
        current_video = videos_list.loc[j][0]

        current_df = pd.read_csv(os.path.join(poses_path,current_video + "_poses_landamarks.csv"))
        current_df['label'] = np.zeros(len(current_df))

        df_annotations_video = df_annotations[df_annotations["video"] == current_video]
        df_annotations_video = df_annotations_video.reset_index()

        for i in range(len(df_annotations_video)):
            current_df.loc[(df_annotations_video["t_start"][i] <= current_df["frame"]) & (
                        current_df["frame"] <= df_annotations_video["t_end"][i]), "label"] = df_annotations_video["id"][
                i]

        df = pd.concat([df, current_df])

    df.to_csv(out_dir, index=False)


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--videos_csv_path", type=str, default="",
                        help="Path to a csv file containing the videos of the dataset")
    parser.add_argument("--path_DS_distribution", type=str, default="",
                        help="Path containig the splits of the dataset in train/val/test")
    parser.add_argument("--path_landmarks", type=str, default="",
                        help="Path containing the folder with the landmarks extracted by mediaPipe")

    parser.add_argument("--out_dir", type=str, default="", help="Path to save the generated file with compacted landmarks")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()
    compact_data_IPNHand(args.videos_csv_path, args.path_DS_distribution, args.path_landmarks, args.out_dir)