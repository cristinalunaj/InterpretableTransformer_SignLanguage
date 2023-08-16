import os.path
import numpy as np
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--path_DS_distribution", type=str, default="", help="Path containig the splits of the dataset in train/val/test")
    parser.add_argument("--path_DS_metadata", type=str, default="", help="Only necessary for IPNHand")
    parser.add_argument("--nameOfDS", type=str, default="WLASL100", help="Name of the dataset to generate", choices=["WLASL100", "IPNHand"])
    parser.add_argument("--out_dir", type=str, default="", help="Path to save the generated CSV files")
    parser.add_argument("--path_landmarks", type=str, default="", help="Path to the files with the landmarks. In the case of WLASL100 it will be the name of "
                                                                       "the folder with the landarmks, for IPNHand is a complete CSV file containing all the landmarks. "
                                                                       "Only the (x,y) coordinates will be extracted")
    return parser


def create_pose_sets(path_pose_features, df_videos, out_path, filterword):
    # open file in write mode
    with open(out_path, 'w') as fp:
        for i, df_row in df_videos.iterrows():
            video_id = str(df_row["video_id"]).zfill(5)
            print("Processing video: ", video_id)
            label_numb = int(df_row["gloss_number"])
            path_landm = os.path.join(path_pose_features, video_id+"_poses_landmarks.csv")
            df_land = pd.read_csv(path_landm, sep=";", header=0)
            df_land = df_land.filter(regex='|'.join(filterword)).dropna(how='all')
            df_land["labels"] = label_numb
            #df_land["video_id"] = video_id
            #array_land = [np.array2string(df_land.values[i], separator=',') for i in range(len(df_land.values))]
            if(i==0):
                fp.write(
                    '%s' % np.array2string(df_land.columns.values, separator=',').replace("\n ", "")[2:-1].replace(
                        "'", "")+"\n")
            for col in df_land.columns:
                # write each item on a new line
                if(col==df_land.columns[-1]):
                    fp.write('"%s"' % np.array2string(df_land[col].values, separator=',').replace("\n ", "").strip()+"\n")
                else:
                    fp.write('"%s",' % np.array2string(df_land[col].values, separator=',').replace("\n ", "").strip())


def create_pose_sets_IPNHand(df, df_metadata, df_annotations, out_path, set2create, num_series=42, granularity=5):
    fea_list = [["x" + str(j), "y" + str(j)] for j in range(int(num_series / 2))]
    flat_fea_list = [item for sublist in fea_list for item in sublist]
    with open(out_path, 'w') as fp:
        for i in range(len(df_annotations)):
            current_video = df_annotations["video"][i]
            current_label = df_annotations["id"][i]
            current_example = df.loc[(df["video"] == current_video) & (df["label"] == current_label) & (
                        df_annotations["t_start"][i] <= df["frame"]) & (df["frame"] <= df_annotations["t_end"][i])]
            current_example = current_example.reset_index(drop=True)
            Nsamples = int(len(current_example) / granularity) + 1
            currentSet = df_metadata.loc[df_metadata["Video Name"]==current_video, "Set"].values[0]
            if (Nsamples <= 0): continue
            #check current set:
            df_land = current_example[flat_fea_list]
            df_land["labels"] = current_example["label"]
            if (i == 0):
                fp.write(
                    '%s' % np.array2string(df_land.columns.values, separator=',').replace("\n ", "")[2:-1].replace(
                        "'", "") + "\n")
            # check if data belongs to current set
            if(set2create == currentSet):
                for col in df_land.columns:
                    # write each item on a new line
                    if (col == df_land.columns[-1]):
                        fp.write('"%s"' % np.array2string(df_land[col].values, separator=',').replace("\n ", "").strip() + "\n")
                    else:
                        fp.write('"%s",' % np.array2string(df_land[col].values, separator=',').replace("\n ", "").strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()

    # WLASL100
    if(args.nameOfDS=="WLASL100"):
        select_column_containing = ["x","y"]
        #path_DS_distribution = "../WLASL/"+args.nameOfDS+"_v0.3.csv"
        #path_pose_features = ".../RESOURCES/WLASL/pose_features_clean"
        out_path = os.path.join(args.out_dir, ('-'.join(select_column_containing)))
        os.makedirs(out_path, exist_ok=True)

        # Join info
        df_distribution = pd.read_csv(args.path_DS_distribution, sep=";", header=0)
        df_train = df_distribution.loc[df_distribution["split"]=="train"]
        df_train = df_train.reset_index(drop=True)
        df_val = df_distribution.loc[df_distribution["split"] == "val"]
        df_val = df_val.reset_index(drop=True)
        df_test = df_distribution.loc[df_distribution["split"] == "test"]
        df_test = df_test.reset_index(drop=True)
        print("Files per set: ")
        print(">>> Train: ", str(len(df_train)))
        print(">>> Val: ", str(len(df_val)))
        print(">>> Test: ", str(len(df_test)))

        create_pose_sets(args.path_landmarks, df_train, os.path.join(out_path, "WLASL100_train.csv"),
                         filterword=select_column_containing)
        create_pose_sets(args.path_landmarks, df_val, os.path.join(out_path, "WLASL100_val.csv"),
                         filterword=select_column_containing)
        create_pose_sets(args.path_landmarks, df_test, os.path.join(out_path, "WLASL100_test.csv"),
                         filterword=select_column_containing)
    elif (args.nameOfDS == "IPNHand"):
        # IPNHand
        for set2create in ["train", "test"]: # train test
            out_filename = "IPNHand_"+set2create+".csv"
            #path_DS_distribution_file = "../RESOURCES/IPNHand/data_MGM/Annot_List.txt"
            #path_DS_metadata = "/.../RESOURCES/IPNHand/metadata.csv"
            #path_pose_features = "/.../RESOURCES/IPNHand/data_MGM/coordinates_frames_labelled.csv"
            #out_path = "/data/IPNHand/own_data"

            # Join info
            df_distribution = pd.read_csv(args.path_DS_distribution)
            df = pd.read_csv(args.path_landmarks)
            df_metadata = pd.read_csv(args.path_DS_metadata)
            create_pose_sets_IPNHand(df, df_metadata, df_distribution, os.path.join(args.out_dir, out_filename),
                                     set2create, num_series=42, granularity=5)











