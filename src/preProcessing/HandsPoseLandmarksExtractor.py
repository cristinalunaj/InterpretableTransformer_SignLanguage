import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from google.protobuf.json_format import MessageToDict
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import argparse


def extract_XYZ(hands, pose, IMAGE_FILES, user, images_video_path, out_path_df, columns, label, padVal=-2):
    video_df = pd.DataFrame([], columns=columns + ["video", "user", "frame", "path", "label"])

    for idx, file in enumerate(IMAGE_FILES):
        df_handsBody_coordinates = pd.DataFrame(padVal * np.ones(shape=(1, len(columns))), columns=columns)
        try:
            # print(file)
            path_img = os.path.join(images_video_path, file)
            # Traditional way
            # image = cv2.imread(path_img)
            # For coding utf-8 filenames
            numpyarray = np.asarray(bytearray(open(path_img, "rb").read()), dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results_p = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Hands
            if results.multi_hand_landmarks is None:
                print("no hands detected")
            else:
                for iVal, hand_handedness in enumerate(results.multi_handedness):
                    handedness_dict = MessageToDict(hand_handedness)
                    handDetected = handedness_dict['classification'][0]["label"]
                    if (handDetected == "Left"):
                        hand2process = "LH"
                    else:
                        hand2process = "RH"
                        # print("Hand ", handDetected)

                    hand_landmarks = results.multi_hand_landmarks[
                        iVal]  # Follow index of results.multi_handedness - NOT THE 'index' OF THE DICT
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y  # (1-y for providing same shape as in image)
                        z = hand_landmarks.landmark[i].z
                        # values+=[x, y, z]
                        df_handsBody_coordinates[hand2process + "x" + str(i)] = x
                        df_handsBody_coordinates[hand2process + "y" + str(i)] = y
                        df_handsBody_coordinates[hand2process + "z" + str(i)] = z

            # Pose
            if results_p.pose_landmarks is None:
                print("no body detected")
            else:
                for i in range(len(results_p.pose_landmarks.landmark)):
                    x = results_p.pose_landmarks.landmark[i].x
                    y = results_p.pose_landmarks.landmark[i].y  # (1-y for providing same shape as in image)
                    z = results_p.pose_landmarks.landmark[i].z
                    df_handsBody_coordinates["Px" + str(i)] = x
                    df_handsBody_coordinates["Py" + str(i)] = y
                    df_handsBody_coordinates["Pz" + str(i)] = z

            video_df = video_df.append(pd.DataFrame([list(df_handsBody_coordinates.values[0]) + [user] + [
                file.split("_")[0], (file.split(".")[0].split("_")[-1]), path_img, label]],
                                                    columns=columns + ["video", "user", "frame", "path", "label"]))
        except Exception:
            with open('logs.txt', 'a') as f:
                print('extract_XYZ() ', images_video_path, ' ', idx + 1, ' ', file, file=f)
    video_df.to_csv(out_path_df, sep=",", header=True, index=False)



def main_extract_WLASL(root_path, classes_df, out_path_directory, hands, pose, columns):
    for video in os.listdir(root_path):  # os.listdir(root_path) #classes_df["video_id"].values
        video = "{:05d}".format(int(video))
        images_video_path = os.path.join(root_path, video)
        if (os.path.isdir(images_video_path)):
            print("Processing ", video, " ...")
        if (os.path.isfile(os.path.join(out_path_directory, video + '_poses_landmarks.csv'))):
            # video was already processed
            continue

        for filename in os.listdir(images_video_path):
            if (filename.split(".")[-1] == "bmp"):
                new_filename = '_'.join(str(x) for x in filename.split(".")[0].split("_")[:-1])
                os.rename(images_video_path + "/" + filename,
                          images_video_path + "/" + new_filename + "_" + filename.split(".")[0].split("_")[-1].zfill(
                              4) + "." + filename.split(".")[1])
        IMAGE_FILES = sorted(os.listdir(images_video_path))
        if (len(classes_df.loc[classes_df["video_id"] == video]) <= 0):
            print("NO GLOSA PARA VIDEO: ", video)
            # Second check:
            if (len(classes_df.loc[classes_df["video_id"] == str(int(video))]) <= 0):
                print("NADA ... NO GLOSA PARA VIDEO: ", video)
                continue
            else:
                label = classes_df.loc[classes_df["video_id"] == str(int(video))]["gloss_name"].values[0]
                user = classes_df.loc[classes_df["video_id"] == str(int(video))]["signer_id"].values[0]
        else:
            label = classes_df.loc[classes_df["video_id"] == video]["gloss_name"].values[0]
            user = classes_df.loc[classes_df["video_id"] == video]["signer_id"].values[0]

        out_path_df = os.path.join(out_path_directory, video + '_poses_landmarks.csv')
        extract_XYZ(hands, pose, IMAGE_FILES, user, images_video_path, out_path_df, columns, label)


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--frames_path", type=str, default="",
                        help="Path containing the videos of the dataset to extract the frames from")
    parser.add_argument("--labels_path", type=str, default="",
                        help="Path containing the videos of the dataset to extract the frames from")
    parser.add_argument("--out_dir", type=str, default="", help="Path to save the generated frames")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()

    #root_path = "/media/acoucheiro/clj/WLASL/raw_frames_WLASL2000"
    #path_labels = "/media/acoucheiro/clj/WLASL/WLASL_v0.3.csv"
    #out_path_directory = "/media/acoucheiro/clj/WLASL/pose_features"
    os.makedirs(args.out_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                        min_detection_confidence=0.5)

    classes_df = pd.read_csv(args.labels_path, sep=";", header=0, dtype={'video_id': str})
    with open('logs.txt', 'a') as f:
        print('NEW RUN ', datetime.now().strftime("%d/%m/%Y %H:%M:%S"), ' errors in this run:', file=f)

    columns1 = [["RHx" + str(i), "RHy" + str(i), "RHz" + str(i)] for i in range(21)]
    columns2 = [["LHx" + str(i), "LHy" + str(i), "LHz" + str(i)] for i in range(21)]
    columns3 = [["Px" + str(i), "Py" + str(i), "Pz" + str(i)] for i in range(33)]
    columns = np.concatenate((columns1, columns2), axis=0)
    columns = np.concatenate((columns, columns3), axis=0)
    columns = [item for sublist in columns for item in sublist]

    main_extract_WLASL(args.frames_path, classes_df, args.out_dir, hands, pose, columns)


