
import os, sys
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from src.trainingUtils.utils import __balance_val_split, __split_of_train_sequence, __log_class_statistics, train_epoch, evaluate
from src.dataloader.czech_slr_dataset import CzechSLRDataset
from src.models.spoter_model_original import SPOTER, SPOTERnoPE
from src.models.BaselineTransformerClassification import BaselineTransformerClassification
from src.models.ExplainabTransformer import ExplainabTransformerwQuery, ExplainabTransformerwSequence
from src.augmentations.gaussian_noise import GaussianNoise



def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="WLASL100_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--n_heads", type=int, default=9,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")
    parser.add_argument("--model2use", type=str, choices=["originalSpoterPE", "originalSpoterNOPE", "baselineTransformer",
                                                          "ownModelwquery", "ownModelwseq"],
                        default="originalSpoterPE",
                        help='Type of model to select for the training. choices=["originalSpoterPE", '
                             '"originalSpoterNOPE", "baselineTransformer","ownModelwquery", "ownModelwseq"]')

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")
    # Landmarks library:
    parser.add_argument("--mediaPipe", type=bool, default=True,
                        help="Determines whether the landmarks were generated using MediaPipe[True] or using VisionAPI[False]")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")


    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    return parser


def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

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
    n_landmarks = int(args.hidden_dim/2) # 54 21 42 21 75

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model
    if(args.model2use=="originalSpoterPE"):
        print("Using model originalSpoter")
        slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim, n_heads=args.n_heads)
    elif (args.model2use == "originalSpoterNOPE"):
        print("Using model originalSpoter WITHOUT POSITIONAL ENCODING")
        slrt_model = SPOTERnoPE(num_classes=args.num_classes, hidden_dim=args.hidden_dim, n_heads=args.n_heads)
    elif(args.model2use=="baselineTransformer"):
        print("Using model baselineTransformer")
        slrt_model = BaselineTransformerClassification(num_classes=args.num_classes, hidden_dim=args.hidden_dim, n_heads=args.n_heads)
    elif(args.model2use=="ownModelwquery"):
        print("Using model ownModelwquery")
        slrt_model = ExplainabTransformerwQuery(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                                                       n_heads=args.n_heads)
    elif(args.model2use=="ownModelwseq"):
        print("Using model ownModelwseq")
        slrt_model = ExplainabTransformerwSequence(num_classes=args.num_classes, hidden_dim=args.hidden_dim,
                                                n_heads=args.n_heads)

    slrt_model.train(True)
    slrt_model.to(device)

    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()
    sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)


    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, mediapipe=args.mediaPipe, n_landmarks=n_landmarks)

    # Validation set
    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path, mediapipe=args.mediaPipe, n_landmarks=n_landmarks)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)


    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        test_set = CzechSLRDataset(args.testing_set_path, mediapipe=args.mediaPipe, n_landmarks=n_landmarks)
        test_loader = DataLoader(test_set, shuffle=True, generator=g)

    else:
        test_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)


    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):
        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            _, _, val_acc = evaluate(slrt_model, val_loader, device, num_classes=args.num_classes)
            slrt_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                print("... Saving checkpoint: out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")
                torch.save(slrt_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING
    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if test_loader:
        for i in range(checkpoint_index+1):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, test_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")


    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
