import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from tensorboardX import SummaryWriter

import sys
import os
from pathlib import Path
from datetime import datetime

from .train import *
from .validate import *
from .evaluate import *
from .evaluate2 import *
from .predict import *
from .models import *
from .optimizer import *

class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    Args:
        optimizer : Optimizer 
            Wrapped optimizer.
        max_decay_steps (int): 
            after this step, we stop decreasing learning rate
        min_learning_rate : float 
            scheduler stopping learning rate decay, value of learning rate must 
            be this value
        power : float 
            The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, min_learning_rate=1e-5, 
                 power=1.0):

        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')

        self.max_decay_steps = max_decay_steps
        self.min_learning_rate = min_learning_rate
        self.power = power
        self.last_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.min_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.min_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):

        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        if self.last_step <= self.max_decay_steps:
            decay_lrs = [
                (base_lr - self.min_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** self.power) +
                self.min_learning_rate for base_lr in self.base_lrs
            ]

            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

def get_optimizer(optimizer, model, params, lr, momentum):
    optimizer = optimizer.lower()
    if optimizer == 'sgd':
        return torch.optim.SGD(params, lr, momentum=momentum)
    elif optimizer == 'nesterov':
        return torch.optim.SGD(params, lr, momentum=momentum, nesterov=True)
    elif optimizer == 'adam':
        return torch.optim.Adam(params, lr)
    elif optimizer == 'amsgrad':
        return torch.optim.Adam(params, lr, amsgrad=True)
    elif optimizer == 'sam':
        base_optimizer = optim.SGD
        return SAM(model.parameters(), base_optimizer, lr=lr, 
                   momentum=momentum)
    else:
        raise ValueError(
            f"{optimizer} currently not supported, please customize your \
            optimizer in compiler.py"
        )

def weighted_average_overlay(pred_dict, overlay_pixels):
    if isinstance(pred_dict, dict):
        key_ls = ["top", "center", "left", "right", "bottom"]
        key_miss_ls = [m for m in pred_dict.keys() if m not in key_ls]
        if len(key_miss_ls) == 0:
            pass
        else:
            assert "Input must be dictionary containing data for centered "\
                   "image and its 4 neighbors. Missed {}"\
                    .format(", ".join(key_miss_ls))
    else:
        assert "Input must be dictionary containing data for centered image "\
               "and its 4 neighbors, including including 'top', 'left', "\
                "'right', and  'bottom'"

    target = pred_dict['center']
    h, w = target.shape
    # top
    if pred_dict['top'] is not None:
        target_weight = np.array(
            [1. / overlay_pixels * np.arange(1, overlay_pixels + 1)] * w
        ).transpose(1, 0)
        comp_weight = 1. - target_weight
        # comp = scores_dict["up"][- overlay_pixs : , : ]
        target[:overlay_pixels, :] = comp_weight * \
            pred_dict['top'][- overlay_pixels:, :] + \
            target_weight * target[:overlay_pixels, :]
    else:
        pass
    # bottom
    if pred_dict['bottom'] is not None:
        target_weight = np.array(
            [1. / overlay_pixels * \
             np.flip(np.arange(1, overlay_pixels + 1))] * w
        ).transpose(1, 0)
        comp_weight = 1. - target_weight
        target[-overlay_pixels:, :] = comp_weight * \
            pred_dict['bottom'][:overlay_pixels, :] + \
            target_weight * target[-overlay_pixels:, :]
    else:
        pass
    # left
    if pred_dict['left'] is not None:
        target_weight = np.array([1. / overlay_pixels * 
                                  np.arange(1, overlay_pixels + 1)] * h)
        comp_weight = 1 - target_weight
        target[:, :overlay_pixels] = comp_weight * \
            pred_dict['left'][:, -overlay_pixels:] + \
            target_weight * target[:, :overlay_pixels]
    else:
        pass
    # right
    if pred_dict['right'] is not None:
        target_weight = np.array(
            [1. / overlay_pixels * \
             np.flip(np.arange(1, overlay_pixels + 1))] * h
        )
        comp_weight = 1 - target_weight
        target[:, -overlay_pixels:] = comp_weight * \
            pred_dict['right'][:, :overlay_pixels] + \
            target_weight * target[:, -overlay_pixels:]
    else:
        pass

    return target

class ModelCompiler:
    def __init__(self, model, buffer, num_classes=3, gpu_devices=[0], 
                 params_init=None, freeze_params=None):
        """
        Compiler of specified model
        Args:
            model : ''nn.Module''
                pytorch model for segmentation
            buffer : int
                distance to sample edges not considered in optimization
            num_classes : int 
                number of output classes based on the classification scheme
            gpu_devices : list 
                indices of gpu devices to use
            params_init : dict object 
                initial model parameters
            freeze_params : list
                list of indices for parameters to keep frozen
        """

        # s3 client
        #self.s3_client = boto3.client("s3")

        # model
        self.gpu_devices = gpu_devices
        self.num_classes = num_classes

        acceptable_torch_versions = [
            "1.4.0", "1.5.0", "1.5.1", "1.6.0", "1.7.0", "1.7.1", "1.8.0", 
            "1.8.1", "1.9.0", "1.9.1", "1.10.0"
        ]

        if (len(self.gpu_devices) > 1 and 
            torch.__version__ in acceptable_torch_versions):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            self.model = model

        self.model_name = self.model.__class__.__name__

        if params_init:
            self.load_params(params_init, freeze_params)

        self.buffer = buffer

        # gpu
        self.gpu = torch.cuda.is_available()

        if self.gpu:
            print('----------GPU available----------')
            # GPU setting
            if gpu_devices:
                torch.cuda.set_device(gpu_devices[0])
                self.model = torch.nn.DataParallel(self.model, 
                                                   device_ids=gpu_devices)
            self.model = self.model.cuda()

        num_params = sum([p.numel() for p in self.model.parameters() 
                          if p.requires_grad])
        print("total number of trainable parameters: {:2.1f}M"\
              .format(num_params / 1000000))

        if params_init:
            print("--------- Pre-trained model compiled successfully ---------")
        else:
            print("--------- Vanilla Model compiled successfully ---------")

    def load_params(self, dir_params, freeze_params):

        inparams = torch.load(dir_params)

        ## overwrite model entries with new parameters
        model_dict = self.model.state_dict()

        if "module" in list(inparams.keys())[0]:
            inparams_filter = {k[7:]: v.cpu() for k, v in inparams.items() 
                               if k[7:] in model_dict}
        else:
            inparams_filter = {k: v.cpu() for k, v in inparams.items() 
                               if k in model_dict}
        model_dict.update(inparams_filter)
        # load new state dict
        self.model.load_state_dict(model_dict)

        # free some layers
        if freeze_params != None:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False


    def fit(self, train_dataset, val_dataset, epochs, optimizer_name, lr_init, 
            lr_policy, criterion, momentum=None, resume=False, 
            resume_epoch=None, aws_bucket=None, aws_prefixout=None, 
            working_dir=".", **kwargs):

        # working_dir = working_dir
        # working_dir = os.getcwd()
        self.checkpoint_dir = os.path.join(working_dir, "checkpoint")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        print('----------------------- Start training -----------------------')
        start = datetime.now()

        # Tensorboard writer setting
        writer = SummaryWriter(working_dir)

        train_loss = []
        val_loss = []
        lr = lr_init
        # lr_decay = lr_decay if isinstance(lr_decay,tuple) else (lr_decay,1)
        optimizer = get_optimizer(
            optimizer_name, self.model, 
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr, momentum
        )

        # initialize different learning rate scheduler
        lr_policy = lr_policy.lower()
        if lr_policy == "StepLR".lower():
            step_size = kwargs.get("step_size", 3)
            gamma = kwargs.get("gamma", 0.98)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

        elif lr_policy == "MultiStepLR".lower():
            milestones = kwargs.get("milestones", 
                                    [15, 25, 35, 50, 70, 90, 120, 150, 200])
            gamma = kwargs.get("gamma", 0.5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma,
            )

        elif lr_policy == "ReduceLROnPlateau".lower():
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.8)
            patience = kwargs.get('patience', 3)
            threshold = kwargs.get('threshold', 0.0001)
            threshold_mode = kwargs.get('threshold_mode', 'rel')
            min_lr = kwargs.get('min_lr', 3e-6)
            verbose = kwargs.get('verbose', True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience, 
                threshold=threshold, threshold_mode=threshold_mode, 
                min_lr=min_lr, verbose=verbose
            )

        elif lr_policy == "PolynomialLR".lower():
            max_decay_steps = kwargs.get('max_decay_steps', 100)
            min_learning_rate = kwargs.get('min_learning_rate', 1e-5)
            power = kwargs.get('power', 0.8)
            scheduler = PolynomialLR(
                optimizer, max_decay_steps=max_decay_steps, 
                min_learning_rate=min_learning_rate,
                power=power
            )

        elif lr_policy == "CyclicLR".lower():
            base_lr = kwargs.get('base_lr', 3e-5)
            max_lr = kwargs.get('max_lr', 0.01)
            step_size_up = kwargs.get('step_size_up', 1100)
            mode = kwargs.get('mode', 'triangular')
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr, 
                step_size_up=step_size_up, mode=mode
            )

        else:
            scheduler = None

        if resume:
            model_state_file = os.path.join(
                self.checkpoint_dir,
                f"{resume_epoch}_checkpoint.pth.tar"
            )
            if aws_bucket and aws_prefixout:
                self.s3_client.download_file(
                    Bucket=aws_bucket, 
                    Key=os.path.join(
                        aws_prefixout, f"{resume_epoch}_checkpoint.pth.tar"
                    ), Filename=model_state_file
                )
                print(f"Checkpoint file downloaded from s3 and saved to \
                      {model_state_file}")
            # Resume the model from the specified checkpoint in the config file.
            if os.path.exists(model_state_file):
                checkpoint = torch.load(model_state_file)
                resume_epoch = checkpoint["epoch"]
                scheduler.load_state_dict(checkpoint["scheduler"])
                self.model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_loss = checkpoint["train_loss"]
                val_loss = checkpoint["val_loss"]
            else:
                raise ValueError(f"{model_state_file} does not exist")

        if resume:
            iterable = range(resume_epoch, epochs)
        else:
            iterable = range(epochs)

        for t in iterable:

            print(f"[{t + 1}/{epochs}]")

            # start fitting
            start_epoch = datetime.now()
            train(train_dataset, self.model, criterion, optimizer, scheduler,
                  gpu=self.gpu, train_loss=train_loss)
            validate(val_dataset, self.model, criterion, self.buffer, 
                     gpu=self.gpu, valLoss=val_loss)

            # Update the scheduler
            if lr_policy in ["StepLR".lower(), "MultiStepLR".lower()]:
                scheduler.step()
                print(f"LR: {scheduler.get_last_lr()}")

            if lr_policy == "ReduceLROnPlateau".lower():
                scheduler.step(val_loss[t])

            if lr_policy == "PolynomialLR".lower():
                scheduler.step(t)
                print(f"LR: {optimizer.param_groups[0]['lr']}")

            # time spent on single iteration
            print('time:', (datetime.now() - start_epoch).seconds)

            # Adjust index and logger to resume status and save checkpoits in 
            # defined intervals.
            # index = t-resume_epoch if resume else t

            writer.add_scalars(
                "Loss",
                {"train_loss": train_loss[t],
                 "val_loss": val_loss[t]},
                t + 1
            )

            checkpoint_interval = 10  # e.g. save every 10 epochs
            if (t + 1) % checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": t + 1,
                        "state_dict": self.model.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    }, os.path.join(
                        self.checkpoint_dir,
                        f"{t + 1}_checkpoint.pth.tar")
                )

        writer.close()

        print(
            f"-------------------------- Training finished in \
                {(datetime.now() - start).seconds}s --------------------------")

    def evaluate(self, eval_dataset, bucket, out_prefix, filename=None):

        print('---------------------- Start evaluation ----------------------')
        start = datetime.now()

        evaluate(eval_dataset, self.model, self.buffer, self.gpu, bucket, 
                 out_prefix, filename)

        print(
            f"-------------------------- Evaluation finished in \
                {(datetime.now() - start).seconds}s --------------------------")

    def evaluate2(self, eval_dataset, out_dir, class_mapping, filename=None):
        """
        Evaluate the accuracy of the model on the provided evaluation dataset.

        Args:
            eval_dataset : DataLoader
                The evaluation dataset
            class_mapping : dict
                A dictionary mapping class indices to class names.
            filename : str 
                The filename to save the evaluation results in the output CSV.
            
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # os.chdir(Path(self.working_dir) / self.out_dir)
        if not filename:
            filename = str(Path(out_dir) / "metrics.csv")

        print("---------------- Start evaluation ----------------")

        start = datetime.now()

        evaluate2(self.model, eval_dataset, self.num_classes, class_mapping, 
                  self.buffer, filename)

        duration_in_sec = (datetime.now() - start).seconds
        print(f"--------- Evaluation finished in {duration_in_sec}s ---------")


    def predict(self, pred_dataset, bucket, out_prefix, mc_samples=None, 
                pred_buffer=None, average_neighbors=False, shrink_buffer=0, 
                filename=""):
        # pred_dataset must be dictionary containing target and all 4 neighbors 
        # if average_neighbors
        # set_trace()
        if average_neighbors == True:
            if isinstance(pred_dataset, dict):
                key_ls = ["top", "center", "left", "right", "bottom"]
                key_miss_ls = [m for m in pred_dataset.keys() 
                               if m not in key_ls]
                if len(key_miss_ls) == 0:
                    pass
                else:
                    assert "pred_dataset must be dictionary containing data "\
                        " for centered image and its 4 neighbors when " \
                        "average_neighbors set to be True. Missed {}"\
                        .format(", ".join(key_miss_ls))
            else:
                assert "pred_dataset must be dictionary containing data for "\
                    "centered image and its 4 neighbors when " \
                    "average_neighbors set to be True, including 'top', " \
                    "'left', 'right', 'bottom'"
        else:
            pass

        print('---------------------- Start prediction ----------------------')
        start = datetime.now()

        _, meta, tile, year = pred_dataset["center"] \
            if isinstance(pred_dataset, dict) else pred_dataset
        name_score = f"{year}_c{tile[0]}_r{tile[1]}{filename}.tif"
        meta.update({
            'dtype': 'int8'
        })
        meta_uncertanity = meta.copy()
        meta_uncertanity.update({
            "dtype": "float64"
        })
        # s3_client = boto3.client("s3")
        prefix_score = os.path.join(out_prefix, "Score")
        new_buffer = pred_buffer - shrink_buffer
        if average_neighbors:
            scores_dict = {
                k: predict(
                    pred_dataset[k], self.model, pred_buffer, gpu=self.gpu, 
                    shrink_pixel=shrink_buffer
                ) if pred_dataset[k]
                else None for k in pred_dataset.keys()
            }
            nclass = len(list(scores_dict['center']))
            overlay_pixs = new_buffer * 2

            for n in range(nclass):
                score_dict = {
                    k: scores_dict[k][n] \
                        if scores_dict[k] else None for k in scores_dict.keys()
                    }
                score = weighted_average_overlay(score_dict, overlay_pixs)
                # write to s3
                score = score[new_buffer: meta['height'] + new_buffer, 
                              new_buffer:meta['height'] + new_buffer]
                score = score.astype(meta['dtype'])

                with rasterio.open(f"{prefix_score}_{n + 1}_{name_score}", "w", 
                                   **meta) as dst:
                    dst.write(score, indexes=1)

                """
                with MemoryFile() as memfile:
                    with memfile.open(**meta) as src:
                        src.write(score, 1)
                    s3_client.upload_fileobj(
                        Fileobj=memfile, Bucket=bucket, 
                        Key=os.path.join(prefix_score + "_{}".format(n + 1), 
                        name_score)
                    )
                """
        # when not average_neighbors
        else:
            if mc_samples:
                scores = mc_predict(pred_dataset, self.model, mc_samples, 
                                    pred_buffer, gpu=self.gpu,
                                    shrink_pixel=shrink_buffer)
                # write score of each non-background classes into s3
                nclass = len(scores)
                prefix_var = os.path.join(out_prefix, "Variance")
                prefix_entropy = os.path.join(out_prefix, "Entropy_MI")
                # subtracting one as we want to ingnore generating results for 
                # boundary class to increase the speed and save space.
                for n in range(nclass - 1):
                    canvas = scores[n][
                        :, new_buffer: meta['height'] + new_buffer,
                        new_buffer: meta['width'] + new_buffer
                    ]

                    mean_pred = np.mean(canvas, axis=0)
                    mean_pred = np.rint(mean_pred)

                    mean_pred = mean_pred.astype(meta['dtype'])

                    with rasterio.open(f"{prefix_score}_{n + 1}_{name_score}", 
                                       "w", **meta) as dst:
                        dst.write(mean_pred, indexes=1)

                    var_pred = np.var(canvas, axis=0)
                    var_pred = var_pred.astype(meta_uncertanity['dtype'])
                    with rasterio.open(f"{prefix_var}_{n + 1}_{name_score}", 
                                       "w", **meta_uncertanity) as dst:
                        dst.write(var_pred, indexes=1)

                    epsilon = sys.float_info.min
                    entropy = -(mean_pred * np.log(mean_pred + epsilon))
                    mutual_info = entropy - np.mean(
                        -canvas * np.log(canvas + epsilon), axis=0
                    )
                    mutual_info = mutual_info.astype(meta_uncertanity['dtype'])
                    with rasterio.open(f"{prefix_entropy}_{n + 1}_{name_score}", 
                                       "w", **meta_uncertanity) as dst:
                        dst.write(mutual_info, indexes=1)

                    """
                    # uploading to AWS s3
                    with MemoryFile() as memfile:
                        with memfile.open(**meta) as src:
                            src.write(canvas, 1)
                        s3_client.upload_fileobj(
                            Fileobj=memfile, Bucket=bucket,
                            Key=os.path.join(prefix_score + "_{}".format(n + 1), 
                            name_score)
                        )
                    """
            else:
                scores = predict(pred_dataset, self.model, pred_buffer, 
                                 gpu=self.gpu, shrink_pixel=shrink_buffer)
                # write score of each non-background classes into s3
                nclass = len(scores)
                for n in range(nclass):
                    canvas = scores[n][new_buffer: meta['height'] + new_buffer, 
                                       new_buffer: meta['width'] + new_buffer]
                    canvas = canvas.astype(meta['dtype'])

                    with rasterio.open(f"{prefix_score}_{n + 1}_{name_score}", 
                                       "w", **meta) as dst:
                        dst.write(canvas, indexes=1)

        print('----------------- Prediction finished in {}s -----------------' \
              .format((datetime.now() - start).seconds))

    # def save_checkpoint(self, bucket, out_prefix, checkpoints):
    #     '''
    #     checkpoints: save last n checkpoint files or list of checkpoint to save
    #     '''
    #     if type(checkpoints) is list:
    #         checkpoint_files = [f"{i}_checkpoint.pth.tar" for i in checkpoints]
    #     else:
    #         checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)]
    #         # sorted by epoch number
    #         checkpoint_files = sorted(
    #             checkpoint_files, key=lambda x: int(re.findall("\d+", x)[0]), 
    #             reverse=True
    #         )[:checkpoints]

    #     for f in checkpoint_files:
    #         file = os.path.join(self.checkpoint_dir, f)
    #         self.s3_client.upload_file(Filename=file, Bucket=bucket,
    #                                    Key=os.path.join(out_prefix, f))
    #         # os.remove(file)
    #     print(f"{checkpoints} checkpoint files saved to s3 at {out_prefix}")

    def save(self, bucket, out_prefix, object="params", filename=""):

        """
        train_loss_dir = "Loss/train_loss"
        val_loss_dir = "Loss/val_loss"
        # upload Loss files to s3
        train_loss = [f for f in os.listdir(train_loss_dir)][0]
        val_loss = [f for f in os.listdir(val_loss_dir)][0]
        train_loss_out = filename + train_loss
        val_loss_out = filename + val_loss
        self.s3_client.upload_file(Filename=os.path.join(train_loss_dir, train_loss),
                                   Bucket=bucket,
                                   Key=out_prefix + f"/{train_loss_dir}/" + train_loss_out)
        self.s3_client.upload_file(Filename=os.path.join(val_loss_dir, val_loss),
                                   Bucket=bucket,
                                   Key=out_prefix + f"/{val_loss_dir}/" + val_loss_out)
        print("Loss files uploaded to s3")
        os.remove(os.path.join(train_loss_dir, train_loss))
        os.remove(os.path.join(val_loss_dir, val_loss))
        """

        if not filename:
            filename = self.model_name
        if object == "params":
            fn_params = "{}_params.pth".format(filename)
            torch.save(self.model.state_dict(), fn_params)

            """
            self.s3_client.upload_file(Filename=fn_params,
                                       Bucket=bucket,
                                       Key=os.path.join(out_prefix, fn_params))
            print("model parameters uploaded to s3!, at ", out_prefix)
            os.remove(fn_params)
            """

        elif object == "model":
            fn_model = "{}.pth".format(filename)
            torch.save(self.model, fn_model)

            """
            self.s3_client.upload_file(Filename=fn_model,
                                       Bucket=bucket,
                                       Key=os.path.join(out_prefix, fn_model))
            os.remove(fn_model)
            """

        else:
            raise ValueError