
import os
import argparse
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.trainingUtils.utils import evaluate
from src.dataloader.czech_slr_dataset import CzechSLRDataset
from src.models.spoter_model_original import SPOTER, SPOTERnoPE
from src.models.BaselineTransformerClassification import BaselineTransformerClassification
from src.models.ExplainabTransformer import ExplainabTransformerwQuery, ExplainabTransformerwSequence



def interpret_WexplainableCoeff(pos_val,name2save, dataset_name="WLASL"):
    pos_val_weighted = pos_val

    if(dataset_name=="IPNHand"):
        # headers IPNHand
        print("Processing headers as IPNHand")
        headers = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7", "x8",
                   "y8", "x9", "y9", "x10", "y10", "x11", "y11", "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15",
                   "x16", "y16", "x17", "y17", "x18", "y18", "x19", "y19", "x20", "y20"]
    else:
        # headers WLASL100
        print("Processing headers as WLASL")
        headers = ["RHx0", "RHy0", "RHx1", "RHy1", "RHx2", "RHy2", "RHx3", "RHy3", "RHx4", "RHy4", "RHx5", "RHy5",
                   "RHx6", "RHy6", "RHx7", "RHy7", "RHx8", "RHy8", "RHx9", "RHy9", "RHx10", "RHy10", "RHx11", "RHy11",
                   "RHx12", "RHy12", "RHx13", "RHy13", "RHx14", "RHy14", "RHx15", "RHy15", "RHx16", "RHy16", "RHx17",
                   "RHy17", "RHx18", "RHy18", "RHx19", "RHy19", "RHx20", "RHy20", "LHx0", "LHy0", "LHx1", "LHy1",
                   "LHx2", "LHy2", "LHx3", "LHy3", "LHx4", "LHy4", "LHx5", "LHy5", "LHx6", "LHy6", "LHx7", "LHy7",
                   "LHx8", "LHy8", "LHx9", "LHy9", "LHx10", "LHy10", "LHx11", "LHy11", "LHx12", "LHy12", "LHx13",
                   "LHy13", "LHx14", "LHy14", "LHx15", "LHy15", "LHx16", "LHy16", "LHx17", "LHy17", "LHx18", "LHy18",
                   "LHx19", "LHy19", "LHx20", "LHy20", "Px0", "Py0", "Px1", "Py1", "Px2", "Py2", "Px3", "Py3", "Px4",
                   "Py4", "Px5", "Py5", "Px6", "Py6", "Px7", "Py7", "Px8", "Py8", "Px9", "Py9", "Px10", "Py10", "Px11",
                   "Py11", "Px12", "Py12", "Px13", "Py13", "Px14", "Py14", "Px15", "Py15", "Px16", "Py16", "Px17",
                   "Py17", "Px18", "Py18", "Px19", "Py19", "Px20", "Py20", "Px21", "Py21", "Px22", "Py22", "Px23",
                   "Py23", "Px24", "Py24", "Px25", "Py25", "Px26", "Py26", "Px27", "Py27", "Px28", "Py28", "Px29",
                   "Py29", "Px30", "Py30", "Px31", "Py31", "Px32", "Py32"]

    x = range(0,len(headers))
    fig = plt.figure(figsize=(30, 10))
    plt.plot(x, pos_val_weighted)
    plt.xticks(x,headers, rotation=90)
    plt.title("Contribution per landmark and coordinate")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # CHECK PER LANDMARK XY
    xs = pos_val_weighted[::2]
    ys = pos_val_weighted[1::2]
    xys = (xs+ys)/2
    fig = plt.figure(figsize=(30, 10))
    plt.plot(range(0,int(len(headers)/2)), xys)
    plt.xticks(range(0,int(len(headers)/2)), headers[::2], rotation=90)
    plt.title("Compact xy - check contribution per landmark")
    plt.grid()
    plt.tight_layout()
    plt.show()
    #Save csv:
    df_data = pd.DataFrame([xys], columns=headers[::2])
    df_data.to_csv("WEIGHTS_SRC_"+name2save+".csv", sep=";", header=True, index=False)





def check_weights(slrt_model, name2save, nameParam="wEnc", dataset_name="WLASL"):
    for k, v in slrt_model.named_parameters():
        print("KEY: ", k)
        print(v)
        if(k == nameParam):
            interpret_WexplainableCoeff(v.flatten().detach().numpy(), name2save, dataset_name)
            break




def getNparams(model):
    print(" #### NETWORK SIZE ####")
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("The model has in total : ", total_params, " parameters (trainable & non-trainable)")
    print("The model has in total : ", trainable_params, " parameters (trainable)")
    print(" ############ ")


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--n_heads", type=int, default=9,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")

    # Landmarks library:
    parser.add_argument("--mediaPipe", type=bool, default=True,
                        help="Determines whether the landmarks were generated using MediaPipe[True] or using VisionAPI[False]")
    parser.add_argument("--model2use", type=str,
                        choices=["originalSpoterPE", "originalSpoterNOPE", "baselineTransformer",
                                 "ownModelwquery", "ownModelwseq"],
                        default="originalSpoterPE",
                        help='Type of model to select for the training. choices=["originalSpoterPE", '
                             '"originalSpoterNOPE", "baselineTransformer","ownModelwquery", "ownModelwseq"]')

    parser.add_argument("--namePE", type=str, default='wEnc',
                        help="name of the positional Encodign layer (For the Query-Class version the name is: 'wEnc', for spoter is 'pos')")

    # Checkpointing
    parser.add_argument("--load_checkpoint", type=str, default="True",
                        help="Determines the path to load weights checkpoints")
    return parser


def test(args):
    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)
    normalize_flag = False
    augmentation_flag = False
    mediaPipe = args.mediaPipe
    n_landmarks = int(args.hidden_dim/2) # 54 21 42 21 75 #21 # 54 42 21
    model2use = args.model2use # originalSpoter baselineTransformer ownModelwquery ownModelwseq
    namePE = args.namePE  # pos wEnc


    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model
    if (args.model2use == "originalSpoterPE"):
        print("Using model originalSpoter")
        slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim, n_heads=args.n_heads)
    elif (args.model2use == "originalSpoterNOPE"):
        print("Using model originalSpoter WITHOUT POSITIONAL ENCODING")
        slrt_model = SPOTERnoPE(num_classes=args.num_classes, hidden_dim=args.hidden_dim, n_heads=args.n_heads)
    elif (args.model2use == "baselineTransformer"):
        print("Using model baselineTransformer")
        slrt_model = BaselineTransformerClassification(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                                                       n_heads=args.n_heads)
    elif (args.model2use == "ownModelwquery"):
        print("Using model ownModelwquery")
        slrt_model = ExplainabTransformerwQuery(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                                                n_heads=args.n_heads)
    elif (args.model2use == "ownModelwseq"):
        print("Using model ownModelwseq")
        slrt_model = ExplainabTransformerwSequence(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                                                   n_heads=args.n_heads)

    #Load weigths:
    slrt_model.load_state_dict(torch.load(args.load_checkpoint).state_dict())
    slrt_model.eval()
    slrt_model.train(False)
    getNparams(slrt_model)
    dataset_name = "IPNHand" if("IPNHand" in args.load_checkpoint) else "WLASL"
    check_weights(slrt_model, args.load_checkpoint.split("/", -2)[-2], namePE, dataset_name)

    slrt_model.to(device)

    ######### PREPARE DATA FOR BEING EVALUATED BY THE NW ################
    print( " #### EVALUATING ... ####")
    test_set = CzechSLRDataset(args.testing_set_path, mediapipe=args.mediaPipe, n_landmarks=n_landmarks)
    test_loader = DataLoader(test_set, shuffle=False, generator=g)

    pred_correct, pred_all, eval_acc = evaluate(slrt_model, test_loader, device, print_stats=True)
    print("ACC: ", str(eval_acc), " (", str(pred_correct), "/", str(pred_all), ")")




if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    test(args)
