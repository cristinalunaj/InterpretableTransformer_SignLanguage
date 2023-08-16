import os, shutil
import time, multiprocessing
import argparse

def frameExtractor(video_name, videos_path, out_path):

    out_name = video_name.split(".")[0]
    out_path_frames_video = os.path.join(out_path, out_name)
    if(not os.path.exists(out_path_frames_video)):
        print(out_name)
        os.makedirs(out_path_frames_video, exist_ok=True)
        os.system("ffmpeg -i "+os.path.join(videos_path, video_name)+" -qscale:v 2 "+out_path_frames_video+"/"+out_name+"_%06d.jpg")


def unwrap_self_extract_frames_ffmpeg(arg, **kwarg):
    """
	Function necessary for doing parallel processing using ffmpeg
	:return: objective function for frames extraction using ffmpeg
	"""
    return frameExtractor(*arg, **kwarg)


def extract_FPS_parallel(videos_list, videos_path, out_path):
    """
    Extract frames in a parallel way using ffmpeg or opencv funcions
    """
    start_time = time.time()
    pool = multiprocessing.Pool(processes=4)  # processes = 7

    pool.map(unwrap_self_extract_frames_ffmpeg, zip(videos_list, [videos_path] * len(videos_list), [out_path] * len(videos_list)))
    pool.close()
    pool.join()
    final_time = (time.time() - start_time)
    print("--- %s Data preparation TIME IN min ---" % (final_time / 60))


def rm_folders(path_files):
    for file in os.listdir(path_files):
        if(os.path.isdir(os.path.join(path_files, file))):
            shutil.rmtree(os.path.join(path_files, file))

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--videos_path", type=str, default="",
                        help="Path containing the videos of the dataset to extract the frames from")
    parser.add_argument("--out_dir", type=str, default="", help="Path to save the generated frames")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()
    #videos_path = "/.../WLASL/WLASL2000"
    #out_path_frames = "/../raw_frames_WLASL2000"
    os.makedirs(args.out_dir, exist_ok=True)
    #frameExtractor(video_name, videos_path, out_path_frames)
    videos_list = os.listdir(args.videos_path)
    extract_FPS_parallel(videos_list, args.videos_path, args.out_dir)
    #rm_folders(videos_path)
