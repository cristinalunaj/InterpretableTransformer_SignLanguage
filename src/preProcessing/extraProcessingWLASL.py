import os.path
import argparse
import pandas as pd
import json


def convertJSONinCSV(file_path, out_path_csv):
    cols = ["gloss_number","gloss_name", "instance_id", "video_id","split", "bbox", "signer_id", "variation_id", "fps", "frame_start", "frame_end","source", "url"]
    complete_df = pd.DataFrame([], columns=cols)
    #Open JSON
    with open(file_path) as ipf:
        content = json.load(ipf)

    cnt_train = 0
    cnt_val = 0
    cnt_test = 0
    gloss_number = 0

    for ent in content:
        gloss = ent['gloss']

        for inst in ent['instances']:
            split = inst['split']
            if split == 'train':
                cnt_train += 1
            elif split == 'val':
                cnt_val += 1
            elif split == 'test':
                cnt_test += 1
            else:
                raise ValueError("Invalid split.")
            #Add data to csv:
            data2include = [inst[col] for col in cols[2::]]
            complete_df = complete_df.append(pd.DataFrame([[gloss_number]+[gloss]+data2include], columns=cols))
        gloss_number+=1
    #Save csv
    complete_df.to_csv(out_path_csv, sep=";", index=False, header=True)
    print('total glosses: {}'.format(len(content)))
    print('total samples: {}'.format(cnt_train + cnt_val + cnt_test))
    print('      > total samples Train: {}'.format(cnt_train))
    print('      > total samples Val: {}'.format(cnt_val))
    print('      > total samples Test: {}'.format(cnt_test))

def create_WLASLsubsets(df_WLASL, out_path_df, k=100):
    df_WLASL_subset = df_WLASL.loc[df_WLASL["gloss_number"]<k]
    print("Selected first k=", len(df_WLASL_subset["gloss_number"].unique()), " glosses")
    df_WLASL_subset.to_csv(out_path_df, sep=";", index=False, header=True)



def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    # Data
    parser.add_argument("--originalJSONfile", type=str, default="",
                        help="Path to the original JSON provided with WLASL to download videos and having labels:WLASL_v0.3.json")

    parser.add_argument("--out_dir", type=str, default="", help="Folder to save the different versions of WLASL with k classes(k=100, k=300, k=1000)")
    return parser



if __name__ == "__main__":
    #file_path = ".../WLASL/start_kit/WLASL_v0.3.json"
    #out_path = ".../WLASL/WLASL_v0.3.csv"
    parser = argparse.ArgumentParser("", parents=[get_args()], add_help=False)
    args = parser.parse_args()

    convertJSONinCSV(args.originalJSONfile, args.originalJSONfile.split(".json")[0]+".csv")
    df_WLASL = pd.read_csv(args.originalJSONfile.split(".json")[0]+".csv", sep=";", header=0)
    create_WLASLsubsets(df_WLASL, os.path.join(args.out_dir, "WLASL100_v0.3.csv"), k=100)
    create_WLASLsubsets(df_WLASL, os.path.join(args.out_dir, "WLASL300_v0.3.csv"), k=300)
    create_WLASLsubsets(df_WLASL, os.path.join(args.out_dir, "WLASL1000_v0.3.csv"), k=1000)


