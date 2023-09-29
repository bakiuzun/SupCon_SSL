"""
PRETRAIN ONLY METER ML DATASET
"""

import argparse
import os
import pprint

import torch
from tqdm import trange
from encoder import (
    SharedNet,
    SharedNetWithPrototypes,
    DualEncoder,
)
from losses import NTXent, PairwiseNTXent
from datasets import (MeterMLDataset,)
from dataset_utils import (LimitedDataset)


def get_args():

    parser = argparse.ArgumentParser("Training script to train a paired encoder")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--use_pairs",
        action="store_true",
        help="whether or not to use pair informations present in the dataset",
    )
    parser.add_argument(
        "--ot_reg",
        action="store_true",
        help="Optimal Transport regularisation"
    )
    parser.add_argument(
        "--use_negative_labels",
        action="store_true",
        help="samples from the negative class are used to make these samples true negatives"
             "in the contrastive meaning of the word",
    )
    parser.add_argument(
        "--exclude_negatives",
        action="store_true",
        help="exclude samples from the Negatives from the datasets",
    )

    parser.add_argument(
        "--scenario",
        default="s1s2",
        choices=["s1s2", "s1", "s2"],
        help="which datasets to use"
    )
    parser.add_argument(
        "--no_augs",
        action="store_true",
        help="whether or not to disable augmentations (better used with use_pairs)",
    )
    parser.add_argument(
        "--clip_sample_values",
        action="store_true",
        help="whether or not to clip the values of the samples to [-25,0] for s1 and [0,1e4] for s2 "
    )

    parser.add_argument(
        "--model",
        choices=["shared", "dual"],
        default="shared",
    )
    parser.add_argument(
        "--regularize_after_block",
        action="store_true",
    )
    parser.add_argument("--data_ratio", type=float, default=1., help="how much of the dataset should be used (default=1.)")
    return parser.parse_args()


def main():
    args = get_args()
    print()

    use_pairs = args.use_pairs
    use_augs = not args.no_augs
    
    regularize_after_block = args.regularize_after_block

    print(f">> args =")
    pprint.pprint(args.__dict__)

    use_augs = not args.no_augs
    augs = RandomAugmentations() if use_augs else T.Resize((72,72))
    if args.scenario == "s1s2":
        dataset1 = MeterMLDataset.train_sentinel1(augs, exclude_negatives=args.exclude_negatives)
        dataset2 = MeterMLDataset.train_sentinel2(augs, exclude_negatives=args.exclude_negatives)
        dataset = TwoDatasets(dataset1, dataset2, use_pairs=args.use_pairs)
        num_datasets = 2
    elif args.scenario == "s1":
        num_datasets = 1
        dataset = MeterMLDataset.train_sentinel1(augs,exclude_negatives=args.exclude_negatives)
    elif args.scenario == "s2":
        num_datasets = 1
        dataset = MeterMLDataset.train_sentinel2(augs, exclude_negatives=args.exclude_negatives)
    else:
        raise NotImplementedError(args.scenario)
    dataset,num_datasets = get_train_dataset(args)
    
    is_single_dataset = num_datasets == 1
    
    assert (not use_augs and use_pairs) or use_augs, f"invalid (use_augs, use_pairs) combination ({use_augs}, {use_pairs})"
    assert (is_single_dataset and use_augs) or not is_single_dataset, f"invalid (is_single_dataset, use_augs) combination ({is_single_dataset}, {use_augs})"
    assert (is_single_dataset and not use_pairs) or not is_single_dataset, "invalid (is_single_dataset, use_pairs) combination ({is_single_dataset}, {use_pairs})"
    assert (args.regularize_after_block and use_pairs) or not args.regularize_after_block, f"invalid (regularize_after_block, use_pairs) combination ({args.regularize_after_block}, {use_pairs})"
    assert (args.regularize_after_block and args.model != "dual") or not args.regularize_after_block, f"regularize_after_block = {args.regularize_after_block}"


    
    # Apply data limit
    dataset = LimitedDataset(
        dataset,
        args.data_ratio,
        shuffle=False, # train_images.json is already shuffled.
    )

    print(f">> Training with {len(dataset)} images")
    if use_pairs:
        print(f">> Using pairs")

    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    n_epochs = args.n_epochs

    print("DEVICE = ",device)
    def save_model(epoch=n_epochs):
        n = 0
        p = "pairs" if use_pairs else "no_pairs"
        get_ident = lambda n: f"model_{args.model}_{epoch}ep_{p}_{n}.pckl"
        ident = get_ident(n)
        while os.path.isfile(ident):
            n += 1
            ident = get_ident(n)
        torch.save(model.state_dict(), f"/share/projects/ottopia/ssl_Baki/poid/{ident}")
        print(f">> save model at {ident}")

    if args.ot_reg:
        assert args.model == "shared"
        model = SharedNetWithPrototypes()
    elif args.model == "shared":
        model = SharedNet(return_block_act=regularize_after_block)
    elif args.model == "dual":
        model = DualEncoder()
    else:
        raise NotImplementedError(args.model)

    model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    NEGATIVE_INDEX = MeterMLDataset.classes_labels["Negative"]

    if use_pairs and not is_single_dataset:
        criterion = PairwiseNTXent(
            blocks=(1 + use_augs) * num_datasets
        )
    else:
        criterion = NTXent(blocks=4 if use_augs and not is_single_dataset else 2)
    criterion.to(device,non_blocking=True)

    print("USE AUGS = ",use_augs)
    print("MODEL = ",model)
    r = trange(n_epochs)
    for epoch in r:
        
        total_loss = 0.0
        for data in loader:

            if args.ot_reg:
                model.normalize_prototypes()

            if is_single_dataset:
                # single dataset scenario
                ((x1, x2), y) = data
                z1 = model(x1.to(device, non_blocking=True))
                z2 = model(x2.to(device, non_blocking=True))
                z = torch.cat((z1,z2))
                negatives = None
                if args.use_negative_labels:
                    negatives = y.to(device, non_blocking=True) == NEGATIVE_INDEX
                    negatives = torch.cat((negatives, negatives))
                loss = criterion(z, negatives)
            else:

                zs = []
                negatives = []
                assert len(data) == num_datasets
                #loss = 0.
                for i in range(len(data)):
                    (x, y) = data[i]

                    if use_augs:

                        #assert len(x1) == 2 and len(x2) == 2, f"invalid length for x1 ({len(x1)}) or x2 ({len(x2)})"
                        assert len(x) == 2, f"invalid length for x ({len(x)})"
                        x1, x2 = x 

                        loss = 0.
                        if torch.isnan(x1).any() or torch.isnan(x2).any():continue
                        ## some images are known to have NaN values

                        if regularize_after_block:
                            z1, b1 = model(x1.to(device, non_blocking=True))
                            z2, b2 = model(x2.to(device, non_blocking=True))

                            loss += (
                                (b1 - b2).pow(2).sum(-1).mean()
                            )
                        else:
                            z1 = model(x1.to(device, non_blocking=True))
                            z2 = model(x2.to(device, non_blocking=True))

                        zs.append(z1)
                        zs.append(z2)

                        if args.use_negative_labels:
                            negatives_1 = y.to(device, non_blocking=True) == NEGATIVE_INDEX
                            negatives.append(negatives_1)
                            negatives.append(negatives_1) # negative indices are the same for both augs

                    else:
                        loss = 0.
                        if torch.isnan(x).any():continue
                        
                        if regularize_after_block:
                            z1, _ = model(x.to(device, non_blocking=True))

                        else:
                            z1 = model(x.to(device, non_blocking=True))

                        zs.append(z1)

                        if args.use_negative_labels:
                            negatives.append(
                                y.to(device, non_blocking=True) == NEGATIVE_INDEX,
                            )

                z = torch.cat(zs)

                negatives = None if len(negatives) == 0 else torch.cat(negatives)
                loss += criterion(z, negatives)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
        if epoch % 100 == 0 and epoch != 0:
            save_model(epoch)

        msg = f"[epoch = {epoch:02}] loss = {total_loss:2.4f}"
        r.set_description(msg)
        print(msg)

    save_model()


if __name__ == "__main__":
    main()
