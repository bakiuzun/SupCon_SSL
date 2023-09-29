import os
import itertools
import argparse
import torch
from tqdm import trange, tqdm
from losses import SupConLoss
import random
import os 
import utils
from encoder import (
    SharedNet,
    DualEncoder,
    resnet18,
    WrappedResnet,
)
from datasets import (
    MeterMLDataset
)
from dataset_utils import (
    TwoDatasets,
    NDatasets,
    LimitedDataset
)
from metrics import ClasswiseAccuracy,Evaluator



def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_mode", 
                        choices=["finetune", "linear"], 
                        default="linear",
                        help="linear will freeze the model and train only the last layer for the classification")
    parser.add_argument(
        "--model",
        choices=["shared", "dual", "resnet18"],
        default="shared",
    )
    parser.add_argument(
        "--dont_include_negatives",
        action="store_true",
        help="whether or not to include negatives in train+evaluation",
    )
    parser.add_argument(
        "--scenario",
        choices=[
            "s1s2",
            "s1s2_fused",
            "s1",
            "s2",
        ],
        default="s1s2",
        help="which datasets to train+evaluate on. \"fused\" means to "
             "fuse representations before classifying",
             )
    #
    parser.add_argument(
        "--data_ratio",
        type=float, default=1.0,
    )

    parser.add_argument(
        "--sup_delta",
        type=float, default=1.0,
    )

    parser.add_argument(
        "--cross_delta",
        type=float, default=1.0,
    )
    ## modified 
    parser.add_argument("--use_sup_con",
                        action="store_true",
                        help="whether or not to use supervised constrastive loss" )
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="seed of the dataset")

    parser.add_argument("--use_augs",
                        action="store_true",
                        help="use augs or not")


    parser.add_argument("--save_path",
                        help="save model path")



    return parser.parse_args()


def fuse(*features):
    return torch.cat(tuple(features), dim=1)


def main():
    args = get_args()

    #torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    n_epochs = args.n_epochs
    eval_mode = args.eval_mode
    is_finetune = eval_mode == "finetune"
    is_fused = args.scenario.endswith("_fused")
    use_augs = args.use_augs
    augs = utils.identity
    
    if use_augs:augs = utils.RandomAugmentations()
    num_datasets = 2
    dataset_names = ["s1", "s2"]

    if args.scenario.startswith("s1s2"):
        dataset_names = ["s1", "s2"]
        dataset_s1 = MeterMLDataset.train_sentinel1(transform=augs) 
        dataset_s2 = MeterMLDataset.train_sentinel2(transform=augs)
        dataset = TwoDatasets(
            dataset_s1,
            dataset_s2,
            use_pairs=args.scenario == "s1s2_fused",
        )
    elif args.scenario == "s1":
        dataset_names = ["s1"]
        num_datasets = 1
        dataset = MeterMLDataset.train_sentinel1(transform=augs)
    elif args.scenario == "s2":
        dataset_names = ["s2"]
        num_datasets = 1
        dataset = MeterMLDataset.train_sentinel2(transform=augs)
    else:
        raise NotImplementedError(args.scenario)

    dataset = LimitedDataset(
        dataset,
        ratio=args.data_ratio,
        seed=args.seed,
        shuffle=False
    )


    print("args dataset = ",args.scenario)

    print(f">> Training with {len(dataset)} ({100 * args.data_ratio:03.02f}%) images on {args.scenario}")
    print(f">> Using model ({args.model}) and evaluation ({eval_mode})")
    print(f">> model = '{args.model_weights}'")

    num_classes = len(Evaluator.classes_labels)
    if args.scenario == "dfc_fused": num_classes = 8

    if args.model == "shared":
        model = SharedNet()
    elif args.model == "dual":
        model = DualEncoder()
    elif args.model == "resnet18":
        model = WrappedResnet()
    else:
        raise NotImplementedError(args.model)
    
    """
    generation of the last layer for the classification
    """
    def gen_lin_classifier():
        lin_classifier = torch.nn.Linear(
            model.output_size * (num_datasets if is_fused else 1),
            num_classes,
        )
        lin_classifier.to(device).train()

        lin_classifier.weight.data.normal_(mean=0.0, std=0.01)
        lin_classifier.bias.data.zero_()
        return lin_classifier

    # if we have multiple modality as s1 s2 without fusion it will create 2 linear
    lin_classifiers = torch.nn.ModuleList([
        gen_lin_classifier()
        for _ in range(num_datasets if (is_finetune or is_finetune == False) and not is_fused else 1)
    ])

    if args.model_weights != "random":
        checkpoint = torch.load(args.model_weights)
        load_result = model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    if eval_mode == "finetune":
        model.train()
    else:
        model.eval()
    model.requires_grad_(is_finetune)


    learning_rate = 0.01
    opt_params = [
        dict(params=lin_classifiers.parameters(), lr=learning_rate),
    ]
    if is_finetune:
        opt_params.append(dict(params=model.parameters(), lr=learning_rate))

    opt = torch.optim.SGD(opt_params)
    


    NEGATIVE_INDEX = Evaluator.classes_labels.index("Negative")
    ignore_index = NEGATIVE_INDEX if args.dont_include_negatives else -100
        
    criterion_cross = torch.nn.CrossEntropyLoss(ignore_index=ignore_index,reduction="mean").to(device)
    criterion_sup = SupConLoss(temperature=0.07,base_temperature=0.07)
    
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )

    val_dataset_s1 = MeterMLDataset.validate_sentinel1()
    val_dataloader_s1 = torch.utils.data.DataLoader(val_dataset_s1, batch_size=32, num_workers=num_workers)
    val_dataset_s2 = MeterMLDataset.validate_sentinel2()
    val_dataloader_s2 = torch.utils.data.DataLoader(val_dataset_s2, batch_size=32, num_workers=num_workers)
    
    val_datasets = {
        "s1": val_dataset_s1,
        "s2": val_dataset_s2,
    }
    val_dataloaders = {
        "s1": val_dataloader_s1,
        "s2": val_dataloader_s2,
    }
    
    # Some model can implement fused representations
    # using fused representations
    print("dataset names = ",dataset_names)
    val_joint_dataset = NDatasets(
        [val_datasets[d] for d in dataset_names], use_pairs=True
    )

   
    val_joint_dataloader = torch.utils.data.DataLoader(
        val_joint_dataset, batch_size=32, num_workers=8,
    )

    if num_datasets == 2:
        for d in dataset_names:
            print(f">> Evaluation {d} ({len(val_dataloaders[d].dataset)})")
    elif args.scenario == "s1":
        print(f">> Evaluation sentinel1 ({len(val_dataset_s1)})")
    elif args.scenario == "s2":
        print(f">> Evaluation sentinel2 ({len(val_dataset_s2)})")


    best_acc = [0,0,0]

    sup_delta = args.sup_delta
    cross_delta = args.cross_delta

    r = trange(n_epochs)


    def eval_dataset(dataloader, lin_classifier):
        model.eval()
        lin_classifiers.eval()
        evaluator = Evaluator(ignore_index=ignore_index)
        print(f">> Evaluating with {len(dataloader)} batches")

        if is_fused:
            for data in tqdm(dataloader):
                X = map(lambda xy: xy[0], data)
                y = next(map(lambda xy: xy[1], data))
                with torch.no_grad():
                    out = lin_classifier(fuse(
                        *(model(x.to(device, non_blocking=True)) for x in X)
                    ))
                labels = y.to(device, non_blocking=True)
                preds = out.argmax(-1)
                evaluator.fit(preds, labels)
            return evaluator.finish()

        for x, y in tqdm(dataloader):
            with torch.no_grad():
                out = lin_classifier(
                    model(x.to(device, non_blocking=True),)
                )
            labels = y.to(device, non_blocking=True)
            preds = out.argmax(-1)
            evaluator.fit(preds, labels)

        return evaluator.finish()

    def eval_all(epoch):
        """
        this method is used to evaluate the model with the Evaluator Class 
        """
        if is_fused:
            acc_fused = eval_dataset(val_joint_dataloader, lin_classifiers[0])
            print(f"{args.scenario} = {acc_fused:03.02f}%")
            if acc_fused > best_acc[0]:
                best_acc[0] = acc_fused
                best_acc[2] = epoch
        elif num_datasets == 1:
            name = dataset_names[0]
            acc_single = eval_dataset(val_dataloaders[name], lin_classifiers[0])
            print(f"{name} = {acc_single:03.02f}%")
        else:
            acc_s1 = eval_dataset(val_dataloader_s1, lin_classifiers[0])
            acc_s2 = eval_dataset(val_dataloader_s2, lin_classifiers[1])
            print(f"s1 = {acc_s1:03.02f}% s2 = {acc_s2:03.02f}%")
   

    def new_validation(epoch,lin_classifier):
        """
        this method is used to evaluate the model using the ClasswiseAccuracy class
        """
        model.eval()

        # track performance
        epoch_losses = torch.Tensor()
        metrics = ClasswiseAccuracy(7)

        for data in tqdm(val_joint_dataloader):
            X = map(lambda xy: xy[0], data)
            y = next(map(lambda xy: xy[1], data))
            labels = y.to(device, non_blocking=True)
            with torch.no_grad():
                out = lin_classifier(fuse(
                        *(model(x.to(device,non_blocking=True) ) for x in X)
                ))

                loss = criterion_cross(out, labels)

                preds = out.argmax(-1)
                epoch_losses = torch.cat([epoch_losses, loss[None].detach().cpu()])
                metrics.add_batch(labels, preds)

                #pbar.set_description(f"Validation Loss:{epoch_losses[-100:].mean():.4}")

        mean_loss = epoch_losses.mean()
                
        val_stats = {
            "validation_loss": mean_loss.item(),
            "validation_average_accuracy": metrics.get_average_accuracy(),
            "validation_overall_accuracy": metrics.get_overall_accuracy(),
            **{
                "validation_accuracy_" + k: v
                for k, v in metrics.get_classwise_accuracy().items()
            },
        }

        ## the best model is saved and the old one deleted 
        """
        if best_acc[0] <  metrics.get_average_accuracy():
            
            path_name = f"{args.save_path}_average_{best_acc[0]* 100:.2f}_epoch_{best_acc[2]}.pth";

            if (epoch != 0):
                os.remove(path_name)
            best_acc[0] =  metrics.get_average_accuracy()
            best_acc[1] =  metrics.get_overall_accuracy()
            best_acc[2] = epoch


            path_name = f"{args.save_path}_average_{best_acc[0]* 100:.2f}_epoch_{best_acc[2]}.pth";

            torch.save(model.state_dict(), path_name)

            print(val_stats)
        """

        # comment this line if you save the best model
        print(val_stats)

    for epoch in r:
        total_loss = 0.


        if is_finetune: model.train()
        lin_classifiers.train()

        for data in dataloader:

            ## NO FUSION here
            if args.scenario == "s1s2":  
                if use_augs != True:
                    x1,y1 = data[0]
                    x2,y2 = data[1]
                    y1 = y1.to(device, non_blocking=True)
                    y2 = y2.to(device, non_blocking=True)

                    feature_z1 = model(x1.to(device, non_blocking=True))
                    feature_z2 = model(x2.to(device, non_blocking=True))
                                 
                    z1 = lin_classifiers[0](feature_z1)
                    z2 = lin_classifiers[1](feature_z2)

                    loss = cross_delta * criterion_cross(z1,y1 ) + cross_delta * criterion_cross(z2,y2)
                    if args.use_sup_con: 

                        ## sup con loss wait a len shape of 3, features are of shape (batch_size,num_features)
                        feature_non_fused_sup_con_s1 = feature_z1.unsqueeze(1)
                        feature_non_fused_sup_con_s2 = feature_z2.unsqueeze(1)
                        loss += sup_delta * criterion_sup(feature_non_fused_sup_con_s1, y1) + sup_delta * criterion_sup(feature_non_fused_sup_con_s2, y2)
                else:
                    ## here we use augmentation 
                    sup_feature_s1,sup_feature_s2,cross_feature_s1,cross_feature_s2,y1,y2 = utils.meterml_return_feature(data=data,model=model,device=device,n_dataset=num_datasets)

                     
                    z1 = lin_classifiers[0](cross_feature_s1.to(device, non_blocking=True))
                    z2 = lin_classifiers[1](cross_feature_s2.to(device, non_blocking=True))

                    loss =  cross_delta * criterion_cross(z1,y1) + cross_delta * criterion_cross(z2,y2) 
                    if args.use_sup_con:
                        loss += sup_delta * criterion_sup(sup_feature_s1, y1) + sup_delta * criterion_sup(sup_feature_s2, y2) 


            elif is_fused:
                if use_augs != True:
                    x1,y1 = data[0]
                    x2,y2 = data[1]
                    y_fused = torch.cat([y1,y2],dim=0)
                    feature_z1 = model(x1.to(device, non_blocking=True))
                    feature_z2 = model(x2.to(device, non_blocking=True))
                                 
                    feature_fused = fuse(feature_z1,feature_z2)
                    z = lin_classifiers[0](feature_fused)

                    loss = cross_delta * criterion_cross(z,y1.to(device,non_blocking=True)) 
                    if args.use_sup_con: 
                        feature_fused_sup_con = torch.cat([feature_z1.unsqueeze(1) ,feature_z2.unsqueeze(1)],dim=0)
                        loss += sup_delta * criterion_sup(feature_fused_sup_con, y_fused.to(device, non_blocking=True))
                   
                else:

                    sup_feature_s1,sup_feature_s2,cross_feature_s1,cross_feature_s2,y1,y2 = utils.meterml_return_feature(data=data,model=model,device=device,n_dataset=num_datasets)
                    y_fused = torch.cat([y1,y2],dim=0)
                    
                    ## we fuse the features of s1 s2 modality of last dim for the cross entropy loss
                    feature_fused_lin_class = fuse(cross_feature_s1,cross_feature_s2)
                    z = lin_classifiers[0](feature_fused_lin_class)

                    loss = cross_delta * criterion_cross(z,y1) 
                    if args.use_sup_con:
                        ## fusion in the batch dim
                        feature_fused_sup_con = torch.cat([sup_feature_s1 ,sup_feature_s2],dim=0)
                        loss += sup_delta * criterion_sup(feature_fused_sup_con, y_fused) 

            else:
                # one modality, S1 or S2
                if use_augs != True:
                    x1, y1 = data
                    features = model(x1.to(device, non_blocking=True))
                    z1 = lin_classifiers[0](features)
                    loss = cross_delta * criterion_cross(z1, y1.to(device, non_blocking=True))
                    if args.use_sup_con:
                        feature_sup_con = features.unsqueeze(1)
                        loss += sup_delta * criterion_sup(feature_sup_con, y1.to(device, non_blocking=True))

                else:
                    sup_feature,cross_feature,y1 = utils.meterml_return_feature(data=data,model=model,device=device,n_dataset=num_datasets)

                    z1 = lin_classifiers[0](cross_feature)
                    loss = cross_delta * criterion_sup(z1, y1) 

                    if args.use_sup_con:
                        loss += sup_delta * criterion_sup(sup_feature, y1) 



            opt.zero_grad()
            loss.backward()
            opt.step()


            total_loss += loss.item()

        msg = f"[epoch = {epoch:02}] loss = {total_loss:2.4f}"
        r.set_description(msg)
        print(msg)
        


        if epoch % 10 == 0 or epoch > 79:
            eval_all(epoch)
            #new_validation(epoch,lin_classifiers[0])

    #new_validation(120,lin_classifiers[0])
    eval_all(epoch)
    
    print(f"BEST ACC = {best_acc[0]} at epoch {best_acc[2]} ")
    print(">> done")


if __name__ == "__main__":
    main()
