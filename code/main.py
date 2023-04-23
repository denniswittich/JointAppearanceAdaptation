import sys, os, time, warnings
from os.path import exists, join as pjoin
import imageio
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dw import eval as dwe
import config, models, tools, losses
import datamanagement as dmng
from datamanagement import EvalDataset
from tools import may_permute, idmap2color as I2C, tensor2numpy as t2n, may_print as mprint


class ExperimentHandler:
    nets: dict  # ----------------------- networks ["seg":segmentation, "gen":generator, "dis":discriminator, ...]
    opts: dict  # ----------------------- corresponding optimizers (same keys as nets)
    class_weights: torch.Tensor  # ------ class weights for the optimizer (used for weighted and adaptive CE loss)
    cf: config.Config  # ---------------- configuration object (see args.py)
    e: int  # --------------------------- current epoch (int)
    e_run: int  # ----------------------- num. of trained epochs in run (different from e if training was interrupted)
    e_es: int  # ------------------------ num. of epochs since model improvement (e.g. increase of validation score)
    F: tools.FolderDict  # ------------- dictionary of folders for saving/loading
    CM: np.ndarray  # ------------------- memory object for confusion matrix
    top_scores: dict  # ----------------- dictionary of best scores
    train_start_time: float  # ---------- training start time
    epoch_start_time: float  # ---------- start time of current epoch
    device: str  # ---------------------- device to use ('cpu' or 'cuda')
    criterion: object  # ---------------- the supervised loss criterion
    once: object  # --------------------- Used to make prints in loop only once

    tra_dataset: Dataset
    tra_data_loader: DataLoader
    ada_tra_dataset: Dataset
    ada_tra_data_loader: DataLoader
    val_dataset: EvalDataset
    val_data_loader: DataLoader
    tes_dataset: EvalDataset
    tes_data_loader: DataLoader
    ada_val_dataset: EvalDataset
    ada_val_data_loader: DataLoader
    ada_tes_dataset: EvalDataset
    ada_tes_data_loader: DataLoader

    # ============================================================================================================ INIT

    def __init__(self, cf: config.Config):
        """Create ExperimentHandler based on configuration object.

        Initializes ExperimentHandler and corresponding attributes. Sets up folder structure and auxiliary variables.
        :param cf: Configuration object
        """
        self.cf = cf
        self.once = tools.Once()
        if cf.VRBS == 0: warnings.filterwarnings("ignore")
        mprint(f'\n\033[31m{cf.OUTPUTS.FOLDER}\033[0m\n', 1, cf.VRBS)
        self.make_output_folders()
        self.store_config()
        self.init_cuda()
        self.init_aux_vars()

        torch.random.manual_seed(cf.RANDOM_SEED)
        np.random.seed(cf.RANDOM_SEED)

    def make_output_folders(self):
        """Create folders for storing results.

        When using folder dicts, the corresponding folders are created on the first usage.
        This prevents creating unused folders.
        """
        ROOT = self.cf.OUTPUTS.FOLDER
        os.makedirs(ROOT, exist_ok=True)
        self.F = tools.FolderDict({
            'confusion_matrices': pjoin(ROOT, 'confusion_matrices'), 'checkpoints': pjoin(ROOT, 'checkpoints'),
            'train_tdom': pjoin(ROOT, 'images/0b_training_tdom'), 'train': pjoin(ROOT, 'images/0_training'),
            'validation': pjoin(ROOT, 'images/1_validation'), 'testing': pjoin(ROOT, 'images/2_testing'),
            'target_dom': pjoin(ROOT, 'images/3_target_dom'), 'target_dom_val': pjoin(ROOT, 'images/3a_target_dom_val'),
            'target_dom_test': pjoin(ROOT, 'images/3b_target_dom_test'),
            'adapted_images': pjoin(ROOT, 'images/4_adapted_images'),
            'aux_generator': pjoin(ROOT, 'images/5_aux_generator'),
        })

    def store_config(self):
        """Stores the full yaml file including inherited attributes to the output folder.

        The stored yaml file may be reused to repeat an experiment.
        """
        now_s = tools.current_datetime_as_str() + '.yaml'
        with open(pjoin(self.cf.OUTPUTS.FOLDER, now_s), 'w+') as f:
            f.write(f"# python main.py {' '.join(sys.argv[1:])}\n")
            f.write(str(self.cf))

    def init_cuda(self):
        """Defines which GPU should be used.

        Switches to CPU if cf.CUDA = -1 or if no GPU is available.
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cf.CUDA)
        if torch.cuda.is_available() and int(self.cf.CUDA) >= 0:
            mprint(f'Using GPU nr. {self.cf.CUDA}: {torch.cuda.get_device_name(0)}', 1, self.cf.VRBS)
            self.device = 'cuda'
            torch.cuda.empty_cache()
        else:
            mprint('Using CPU only!', 1, self.cf.VRBS)
            self.device = 'cpu'

    def init_aux_vars(self):
        """Sets up all auxiliary variables.

        Auxiliary variables are used for computing/storing metrics, tracking epoch, compute weighted loss.
        Note that the confusion matrix (self.CM) is reused multiple times.
        """
        self.e, self.e_es = -1, -1
        self.train_start_time = -1.0
        self.class_weights = torch.ones(self.cf.DATA.NCLS, dtype=torch.float32, device=self.device)

        self.CM = np.zeros([self.cf.DATA.NCLS, self.cf.DATA.NCLS], np.uint32)
        self.top_scores = {'training': 0.0, 'validation': 0.0, 'testing': 0.0, 'testing_E': 1.0,
                           'target_dom_val': 0.0, 'target_dom_test': 0.0, 'target_dom_val_E': 1.0, }
        self.nets, self.opts = {}, {}

    # =============================================================================================== EXPERIMENT SETUPS

    def prepare_datasets(self):
        """ Sets up datasets and data loaders.

        Setup datasets and data loaders for training, validation, testing and target domain.
        The respective sets have to be defined in the config, otherwise they will be skipped.
        """

        cf = self.cf
        DL_v = partial(DataLoader, batch_size=cf.TRAIN.BTSZ, shuffle=False)  # --- Used for val/test sets
        DL_t = partial(DataLoader, batch_size=cf.TRAIN.BTSZ, shuffle=True, num_workers=cf.TRAIN.NUM_WK,
                       prefetch_factor=cf.TRAIN.PREFETCH_FACTOR, persistent_workers=True, pin_memory=True)  # --- train

        if cf.TRAIN.SD_SET:
            self.tra_dataset = dmng.prepare_training_dataset(cf, cf.SD, cf.TRAIN.SD_SET)
            self.tra_data_loader = DL_t(dataset=self.tra_dataset)
        if cf.EVALUATION.SD_VAL_SET:
            self.val_dataset = EvalDataset(cf, cf.SD, cf.EVALUATION.SD_VAL_SET)
            self.val_data_loader = DL_v(dataset=self.val_dataset)
        if cf.EVALUATION.SD_TEST_SET:
            self.tes_dataset = EvalDataset(cf, cf.SD, cf.EVALUATION.SD_TEST_SET)
            self.tes_data_loader = DL_v(dataset=self.tes_dataset)

        if not hasattr(cf, 'TD'): return

        if cf.TRAIN.TD_SET:
            self.ada_tra_dataset = dmng.prepare_training_dataset(cf, cf.TD, cf.TRAIN.TD_SET)
            self.ada_tra_data_loader = DL_t(dataset=self.ada_tra_dataset)
        if cf.EVALUATION.TD_VAL_SET:
            self.ada_val_dataset = EvalDataset(cf, cf.TD, cf.EVALUATION.TD_VAL_SET)
            self.ada_val_data_loader = DL_v(dataset=self.ada_val_dataset)
        if cf.EVALUATION.TD_TEST_SET:
            self.ada_tes_dataset = EvalDataset(cf, cf.TD, cf.EVALUATION.TD_TEST_SET)
            self.ada_tes_data_loader = DL_v(dataset=self.ada_tes_dataset)

    def make_supervised_loss(self, initial_weights: list = None):
        """Setup loss according to the configuration and sets initial weights for classes if provided.

        All loss functions are called with logits: loss = L(logits, labels).
        Weights are initialized by ones if no initial weights are provided.
        :param initial_weights: Initial class weights (optional), given as list of tuples [(class ID, weight), ...]
        """
        cf = self.cf
        ignore_index = cf.DATA.IGNORE_INDEX
        l_type = cf.TRAIN.LOSS.TYPE
        n_cls = cf.DATA.NCLS

        if l_type == 'focal':
            cr = losses.FocalCrossEntropyLoss(n_cls=n_cls, gamma=cf.TRAIN.LOSS.FCL_GAMMA, ignore_index=ignore_index)
        elif l_type == 'dice':
            cr = losses.DiceLoss(n_cls=n_cls, ignore_index=ignore_index)
        elif l_type == 'mae':
            cr = losses.SoftmaxMaeLoss(device=self.device, n_cls=n_cls, ignore_index=ignore_index)
        elif l_type == 'ce':
            cr = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.class_weights)
        elif l_type == 'ace':
            cr = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.class_weights)
        elif l_type == 'sce':
            cr = losses.SampledCrossEntropyLoss(n_cls=n_cls, ignore_index=ignore_index)
        else:
            raise NotImplementedError(f"The loss type {l_type} is not implemented")

        mprint(f"Using {l_type.upper()}-loss", 1, cf.VRBS)
        self.criterion = cr

        if initial_weights is not None:  # ------------------------ may set initial weights
            for c, weight in initial_weights:
                self.class_weights[c] *= weight
            mprint(f'(Initial) weights set to: {self.class_weights}', 1, cf.VRBS)

    def main_metric_from_cm(self):
        """Compute and return the main metric.

        Metric depends on C.EVALUATION.METRIC ('mF1', 'OA').
        Uses an additional repository to calculate metrics.
        """
        main_metric = self.cf.EVALUATION.METRIC

        if np.sum(self.CM) == 0:
            return 0.0
        elif main_metric == 'mF1':
            return dwe.get_confusion_metrics(self.CM)['mf1'].item()
        elif main_metric == 'OA':
            return (np.trace(self.CM) / np.sum(self.CM)).item()
        else:
            raise NotImplementedError(f"The metric {main_metric} is not implemented!")

    def load_checkpoint(self):
        """Loads a checkpoint.

        If LOAD_FROM is specified in the config, the models in the respective checkpoint will be used.
        Else the 'latest.pt' model will be used if it exists (e.g. for continuing training)
        If 'latest.pt' does not exist, the models from LOAD_INIT will be used (e.g. for domain adaptation)
        or default initializations are used (training from scratch or pre-trained weights).
        """

        load = torch.load if torch.cuda.is_available() else partial(torch.load, map_location='cpu')
        init_model = self.cf.CHECKPOINTS.LOAD_INIT
        load_model = self.cf.CHECKPOINTS.LOAD_FROM

        if load_model:  # --------------------------------------- Case 1: Explicit model given by LOAD_FROM
            assert exists(load_model), f"Checkpoint cf.CHECKPOINTS.LOAD_FROM does not exist: {load_model}"
            print('Loading seg model from cf.CHECKPOINTS.LOAD_FROM: ', load_model)
            return load(load_model)
        else:
            checkpoint_file = pjoin(self.F['checkpoints'], 'latest.pt')
            if exists(checkpoint_file):  # ----------------------- Case 2: Loading a native checkpoint
                print('Loading native checkpoint:', checkpoint_file)
                return load(checkpoint_file)
            else:
                out_checkpoint = {'epoch': self.e, 'es_epochs': self.e_es, 'top_scores': self.top_scores}
                if not init_model:  # ----------------------- Case 3: Training from scratch / pre-trained weights
                    print('No checkpoint loaded!')
                    return None
                else:  # ------------------------------------ Case 4: Initializing from other checkpoint (ONLY SEG/GEN)
                    assert exists(init_model), f"Checkpoint cf.CHECKPOINTS.LOAD_INIT does not exist: {init_model}"
                    print('Loading init checkpoint: ', init_model)

                    in_chkpt = load(init_model)
                    keys = in_chkpt.keys()

                    if 'seg_state_dict' in keys:
                        print('Loading seg model from ', init_model)
                        in_seg_data = in_chkpt['seg_state_dict']
                        out_checkpoint['seg_state_dict'] = in_seg_data
                    if 'seg_optimizer_state_dict' in keys:
                        print('Loading segmentation model optimizer from ', init_model)
                        print('Initializing optimizer from cf.CHECKPOINTS.LOAD_FROM')
                        out_checkpoint['seg_optimizer_state_dict'] = in_chkpt['seg_optimizer_state_dict']

                    return out_checkpoint

    def restore_checkpoint(self, checkpoint: dict):
        """Initializes network(s) from a checkpoint.

        If weights for networks are missing or incompatible, they will be skipped.
        :param checkpoint: The loaded checkpoint file to be restored.
        """

        cf = self.cf
        if checkpoint:
            strict = cf.CHECKPOINTS.LOAD_STRICT
            for name in self.nets.keys():
                shape_mismatch = False
                if cf.CHECKPOINTS.LOAD:
                    try:
                        state_dict_to_load = checkpoint[f'{name}_state_dict']
                        target_state_dict = self.nets[name].state_dict()
                        target_keys = target_state_dict.keys()
                        to_drop = []
                        for k, v in state_dict_to_load.items():
                            if k not in target_keys:
                                print(f"Parameter {k} is missing")
                                continue
                            if target_state_dict[k].shape != v.shape:
                                print(f"Removing param set {k} due to size mismatch")
                                to_drop.append(k)
                                shape_mismatch = True
                        for k in to_drop:
                            state_dict_to_load.pop(k)
                        self.nets[name].load_state_dict(state_dict_to_load, strict=strict)
                    except ValueError:
                        print(f'Network {name} incompatible')
                    except KeyError:
                        print(f'Network {name} not in checkpoint')
                if cf.CHECKPOINTS.LOAD_OPT and not shape_mismatch:
                    try:
                        state_dict_to_load = checkpoint[f'{name}_optimizer_state_dict']
                        self.opts[name].load_state_dict(state_dict_to_load)
                    except ValueError:
                        print(f'Optimizer for {name} incompatible')
                    except KeyError:
                        print(f'Optimizer for {name} not in checkpoint')

            self.e = checkpoint['epoch']
            self.e_es = checkpoint['es_epochs']

            stored_top_scores = checkpoint['top_scores']
            for entry in stored_top_scores.keys():
                self.top_scores[entry] = stored_top_scores[entry]

            mprint(f"Restored checkpoint from epoch {self.e}. Top scores: {self.top_scores}", 1, cf.VRBS)

    # ==================================================================================================== NETWORK-INIT

    def init_network(self, net_name: str, make_optimizer: bool = True):
        """Initializes a network and (optionally) its optimizer.

        Used to generate network based on configuration file.
        :param net_name: Network type (e.g. "seg" -> segmentation network).
        :param make_optimizer: Creates the optimizer for the network (not required for evaluation).
        """

        assert net_name in ['seg', 'ap_ad', 'ap_dis', 'aux_gen', 'rp_dis', ], f"Network type {net_name} not supported!"
        net = models.model_from_config(net_name, self.cf).to(self.device)
        self.nets[net_name] = net
        if not make_optimizer: return net

        # --------------------------------------------------------------------- Setup optimizer:
        types_names = {'seg': 'SEG', 'ap_ad': 'AP_AD', 'ap_dis': 'AP_DIS', 'aux_gen': 'AUX_GEN', 'rp_dis': 'RP_DIS'}
        hprm = getattr(self.cf.TRAIN, types_names[net_name])
        btsz = self.cf.TRAIN.BTSZ
        prms = net.parameters()

        if hprm.OPTIM == 'sgd':
            opt = optim.SGD(prms, lr=hprm.LR, momentum=hprm.BETA1, weight_decay=hprm.WDEC)
            t = f"Opt. {net_name}: SGD: lr {hprm.LR} M {hprm.BETA1} [wdec={hprm.WDEC}, bs={btsz}]"
        elif hprm.OPTIM == 'adam':
            opt = optim.Adam(prms, lr=hprm.LR, betas=(hprm.BETA1, hprm.BETA2), weight_decay=hprm.WDEC)
            t = f"Opt. {net_name}: ADAM: lr {hprm.LR} B1/2 {hprm.BETA1}/{hprm.BETA2} [wdec={hprm.WDEC}, bs={btsz}]"
        else:
            raise NotImplementedError(f"Optimizer {hprm.OPTIM} is not supported")

        mprint(t, 1, self.cf.VRBS)
        self.opts[net_name] = opt

        return net, opt

    # ======================================================================================================= AUXILIARY

    def print_training_progress(self):
        """Print information about training progress (epoch, early stopping, runtime, ETA).

        Nothing is printed if verbosity is set to 0.
        """
        cf = self.cf
        if cf.VRBS == 0: return
        if self.train_start_time < 0:  # Setup at first call of the function
            self.train_start_time = self.epoch_start_time = time.time()
            self.e_run = -1

        self.e_run += 1
        run_time = time.time() - self.train_start_time
        it_time = (time.time() - self.epoch_start_time) / cf.TRAIN.IT_P_EP
        self.epoch_start_time = time.time()
        eta_sek = int(run_time / self.e_run * (cf.TRAIN.N_EP_MAX - self.e + 1)) if self.e_run > 0 else 0
        eta_hr, eta_min, eta_sec = tools.sek2hms(eta_sek)

        print(f"\n[{tools.current_datetime_as_str()} / {int(run_time // 60)} min]\
         epoch {self.e}/{cf.TRAIN.N_EP_MAX}, erl.st. cnt. {self.e_es}/{cf.TRAIN.EARLY_STOPPING_EP}\
        , estimated remaining training time is {eta_hr}:{eta_min} ({int(it_time * 1000)} ms/batch)")

    def may_augment_radiometric(self, images, inplace):
        """Performs random radiometric augmentation if cf.AUG.RADIO_SCALE > 0.

        Each channel is multiplied and shifted by a random value, drawn from two normal distributions.
        The augmentation strength (width of the normal distributions) is controlled by cf.AUG.SCALE.
        :param images: Batch of images to augment. Torch tensor with shape [BCHW].
        :param inplace: Whether to perform the augmentation inplace.
        """
        cf = self.cf
        bs, nc, _, _ = images.shape

        if cf.AUG.RADIO_SCALE <= 0.0:
            return None if inplace else torch.clone(images)
        else:
            result = images if inplace else torch.clone(images)
            dev = images.device
            result *= (torch.randn(bs, nc, 1, 1, device=dev) * cf.AUG.RADIO_SCALE + 1.0).float()
            result += (torch.randn(bs, nc, 1, 1, device=dev) * cf.AUG.RADIO_SCALE).float()
            return result

    def may_update_learning_rate(self):
        """Update learning rates of any network for which LR_DEC < 1.0.

        Performs decay of learning rate: LRe = LR*LR_DEC**e
        """

        TRP = self.cf.TRAIN
        names_configs = [
            ('seg', TRP.SEG if hasattr(TRP, 'SEG') else None),
            ('ap_ad', TRP.AP_AD if hasattr(TRP, 'AP_AD') else None),
            ('ap_dis', TRP.AP_DIS if hasattr(TRP, 'AP_DIS') else None),
            ('aux_gen', TRP.AUX_GEN if hasattr(TRP, 'AUX_GEN') else None),
            ('rp_dis', TRP.RP_DIS if hasattr(TRP, 'RP_DIS') else None),
        ]

        for name, conf in names_configs:
            if name in self.opts.keys() and conf.LR_DEC < 1:
                lr = conf.LR * conf.LR_DEC ** self.e
                for g in self.opts[name].param_groups: g['lr'] = lr
                mprint(f"Learning rate of {name} set to {lr}", 1, self.cf.VRBS)

    def may_update_weights(self, cm):
        """ Update the class weights based on class metrics if cf.TRAIN.LOSS.TYPE == 'ace' (adaptive cross entropy).

        :param cm: Either a confusion matrix or a path to a stored one. Based on this matrix the weights are calculated.
        """

        if self.cf.TRAIN.LOSS.TYPE != 'ace': return

        cm_to_use = np.load(cm) if cm is str else cm
        f1s = dwe.get_confusion_metrics(cm_to_use)['f1s']
        np.nan_to_num(f1s, copy=False, nan=0.0)
        delta_f1s = f1s - np.mean(f1s)

        self.class_weights *= 0.0
        self.class_weights += 1.0
        self.class_weights -= torch.tensor(delta_f1s, dtype=torch.float32, device=self.device)
        self.class_weights.pow_(self.cf.TRAIN.LOSS.ACE_POW)

        mprint(f'  F1 scores: {f1s}\n  new weights for ace: {t2n(self.class_weights)}', 2, self.cf.VRBS)

    def all2eval(self):
        """Puts all networks to evaluation mode.

        Nothing is printed if verbosity is < 2.
        """

        mprint("Putting networks to evaluation mode", 2, self.cf.VRBS)
        for net in self.nets.values(): net.eval()

    def all2train(self):
        """Puts all networks to training mode.

        Nothing is printed if verbosity is < 2.
        """

        mprint("Putting networks to training mode", 2, self.cf.VRBS)
        for net in self.nets.values(): net.train()

    def print_num_params(self):
        """Prints the number of parameters for all initialized networks.

        Nothing is printed if verbosity is set to 0.
        """

        if self.cf.VRBS > 0:
            for name in self.nets.keys():
                net = self.nets[name]
                params = list(net.parameters())
                pp = np.sum([np.prod(list(P.size())) for P in params])
                print(f'Model {name} has {pp} prms in {len(params)} vars ({int(pp * 4 / 1000 / 1000 * 10) / 10} MB)')

    def check_finished(self):
        """Checks if training is finished.

        This function should be called after restoring a checkpoint and before each epoch to check
        if training is already finished. Considers max epochs and early stopping. The function returns
        True if training is finished and False otherwise.
        """

        return (0 < self.cf.TRAIN.N_EP_MAX <= self.e) or (0 < self.cf.TRAIN.EARLY_STOPPING_EP <= self.e_es)

    # ========================================================================================================= WRITING

    def may_save_model(self, f_name: str, single_net: str = None):
        """Saves the current model status if cf.CHECKPOINTS.SAVE is True.

        If single_net is None, all models and optimizers are saved.
        If single_net is specified, only this model will be saved (e.g. for deployment)

        :param f_name: Name of file to write to (is appended to the 'checkpoint' folder)
        :param single_net: Name of model to save or None to save the full state (default is None)
        """

        cf = self.cf
        if not cf.CHECKPOINTS.SAVE:
            mprint("Skipping to save network(s)", 2, cf.VRBS)
            return

        if single_net is None:
            save_dict = {'epoch': self.e, 'es_epochs': self.e_es, 'top_scores': self.top_scores}
            for name in self.nets.keys():
                save_dict[f'{name}_state_dict'] = self.nets[name].state_dict()
                if name in self.opts.keys():
                    save_dict[f'{name}_optimizer_state_dict'] = self.opts[name].state_dict()
        else:
            save_dict = {f'{single_net}_state_dict': self.nets[single_net].state_dict()}

        mprint(f"saving network(s) to {f_name}", 1, cf.VRBS)
        torch.save(save_dict, pjoin(self.F['checkpoints'], f_name))

    def save_img_dict(self, img_dict, path):
        """Saves images in an image dict to drive.

        If cf.OUTPUTS.SAVE_SINGLE, the images are horizontally stacked.
        If cf.OUTPUTS.GRID > 0, an grid is superimposed on the images.
        :param img_dict: A dictionary mapping from names to images
        :param path: The path to write the images to, including base-filename. Suffix and extension is added here
        """
        cf, f_ext = self.cf, self.cf.OUTPUTS.IMAGE_EXT
        tools.may_overlay_grid(img_dict, cf.OUTPUTS.GRID)

        if cf.OUTPUTS.SAVE_SINGLE:
            images = list(img_dict.values())
            imageio.imsave(f'{path}.{f_ext}', np.hstack(tuple(images)))
        else:
            for name, image in img_dict.items():
                imageio.imsave(f'{path}-{name}.{f_ext}', image)

    def save_training_samples(self, images, predictions, labels, denorm_dom, suffix='', folder='train'):
        """ Save the first (max=4) training sample in a batch of images.

        :param images: Raw batch of images (torch.Tensor NCHW)
        :param predictions: Raw batch of predictions (torch.Tensor NHW)
        :param labels: Reference labels (torch.Tensor NHW)
        :param denorm_dom: The domain code used for undoing the image normalization
        :param suffix: A string that is appended to the file name
        :param folder: The folder to write the images to
        """
        for b in range(min(images.shape[0], 4)):
            img_dict = tools.denorm4print(t2n(images[b]), denorm_dom, input_order="CHW")
            img_dict['prediction'] = I2C(t2n(predictions)[b, :, :].astype(int), self.cf.SD)
            if labels is not None:
                img_dict['reference'] = I2C(t2n(labels)[b, :, :].astype(int), self.cf.SD)
            self.save_img_dict(img_dict, pjoin(self.F[folder], f'ep{self.e}-{b}{suffix}'))

    def save_images(self, f_name, images, names, denorm_dom, predictions=None, labels=None, input_order='HWC'):
        """Saves a batch of images with corresponding predictions and reference labels.

        :param f_name: Name of folder to store images to (will be created under the 'images' folder)
        :param images: An iterable of images of shape BCHW (channels first)
        :param names: A list of patch names [str,]
        :param denorm_dom: The domain code used for undoing the image normalization
        :param predictions: The predicted label maps as [N x [h x w]] list of arrays (optional)
        :param labels: The reference label maps as [N x [h x w]] list of arrays (optional)
        :param input_order: The order of dimensions of each image
        """
        for i, patch_name in enumerate(names):
            img_dict = tools.denorm4print(np.copy(images[i]), denorm_dom, input_order=input_order)
            if predictions is not None:
                predictions[i][labels[i] == self.cf.DATA.IGNORE_INDEX] = self.cf.DATA.IGNORE_INDEX
                img_dict['prediction'] = I2C(predictions[i][:, :].astype(int), self.cf.SD)
            if labels is not None:
                img_dict['reference'] = I2C(labels[i].astype(int), self.cf.SD)
            self.save_img_dict(img_dict,
                               pjoin(self.F[f_name], f'{patch_name}-{"-".join([k for k in img_dict.keys()])}'))

    def may_save_target_dom_samples(self):
        """This function saves image and prediction for a sample from target domain if cf.OUTPUTS.SAVE_TDOM_TRAIN.

        Uses an augmented image from TD training dataset.
        """
        if self.cf.OUTPUTS.SAVE_TDOM_TRAIN:
            seg = self.nets['seg']
            batch = next(iter(self.ada_tra_data_loader))
            images = batch['image'].to(self.device)
            self.may_augment_radiometric(images, True)
            with torch.no_grad():
                logits = seg(images)
                predictions = torch.argmax(logits, 1)
            self.save_training_samples(images, predictions, None, self.cf.TD, folder='train_tdom', suffix='-TD')

    # ================================================================================================== MAIN PROTOCOLS

    def train_seg(self):
        """Training of segmentation network on the source domain (cf.SD).

        If defined, the validation set cf.EVALUATION.SD_VAL_SET is used for early stopping.
        """

        seg, seg_opt = self.init_network('seg')

        self.print_num_params()
        self.make_supervised_loss()
        self.restore_checkpoint(self.load_checkpoint())
        if self.check_finished(): return
        self.prepare_datasets()

        while not self.check_finished():
            self.e += 1
            self.e_es += 1
            self.may_update_learning_rate()
            self.print_training_progress()

            self.all2train()
            for i, batch in enumerate(self.tra_data_loader):
                if not i % 50: print(f'\r {i}/{self.cf.TRAIN.IT_P_EP}', end='', flush=True)

                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['idmap'].to(self.device, non_blocking=True)
                self.may_augment_radiometric(images, inplace=True)

                sup_loss = self.criterion(seg(images), labels)
                tools.update_network(seg_opt, sup_loss)

            self.all2eval()
            self.quick_eval_seg_on_training_batch()
            self.may_update_weights(self.CM)
            self.evaluate_segmentation_on_subset('validation')
            self.evaluate_segmentation_on_subset('testing')
            self.may_save_model('latest.pt')

    def eval_seg(self):
        """Evaluation of a segmentation network on the source-domain (cf.SD).

        cf.CHECKPOINTS.LOAD_FROM should be set to the path of the checkpoint to load.
        Function may be also used to evaluate an adapted classifier in which case cf.SD may be the target domain.
        """

        self.init_network('seg', make_optimizer=False)
        self.print_num_params()
        self.restore_checkpoint(self.load_checkpoint())
        self.prepare_datasets()
        self.all2eval()
        self.evaluate_segmentation_on_subset('validation')
        self.evaluate_segmentation_on_subset('testing')

    def train_da(self):
        """Domain adaptation (Appearance adaptation, representation transfer, entropy minimization).

        The appearance adaptation is done by training a network that adapts images from source to target domain.
        The classification network is trained on adapted images from source domain with original labels.
        """

        cf, DA = self.cf, self.cf.DA
        c0 = torch.zeros(1, device=self.device)
        L_gen = L_rep = L_sup_s = L_sup_s2t = c0
        # ----------------------------------------------------------------- Initialise networks
        img_dis = rep_dis = None
        seg, seg_opt = self.init_network('seg')
        if DA.DO_APPA:
            appa_net, appa_opt = self.init_network('ap_ad')
            img_dis, dis_opt = self.init_network('ap_dis')
            if cf.DA.AUX_GEN:
                aux_gen, aux_gen_opt = self.init_network('aux_gen')
        if DA.DO_REPA: rep_dis, rep_dis_opt = self.init_network('rp_dis')
        # ----------------------------------------------------------------- Load checkpoint, setup loss & datasets
        self.restore_checkpoint(self.load_checkpoint())
        if self.check_finished(): return
        self.print_num_params()
        self.make_supervised_loss()
        self.prepare_datasets()
        # ----------------------------------------------------------------- Training loop
        while not self.check_finished():
            torch.cuda.empty_cache()
            self.e += 1
            self.e_es += 1 if self.e >= cf.EVALUATION.MIN_ENT_ITER else 0
            self.may_update_learning_rate()
            self.print_training_progress()

            self.all2train()
            if DA.REQUIRES_SD:
                source_iterator = iter(self.tra_data_loader)
            for i, td_batch in enumerate(self.ada_tra_data_loader):
                if not i % 50: print(f'\r {i}/{cf.TRAIN.IT_P_EP}', end='', flush=True)
                if DA.REQUIRES_SD:
                    s_batch = next(source_iterator)
                    s_imgs_raw = s_batch['image'].to(self.device, non_blocking=True)
                    s_lbls = s_batch['idmap'].to(self.device, non_blocking=True)
                    s_imgs = self.may_augment_radiometric(s_imgs_raw, inplace=False)

                t_imgs_raw = td_batch['image'].to(self.device, non_blocking=True)
                t_imgs = self.may_augment_radiometric(t_imgs_raw, inplace=False)

                # ==================================================== TRAIN (AA-NET &) SEGMENTATION (& AUX GENERATOR)
                tools.may_change_requires_grad(img_dis, cf.DA.DO_APPA, new_value=False)
                tools.may_change_requires_grad(rep_dis, cf.DA.DO_REPA, new_value=False)

                loss = torch.clone(c0)
                if DA.DO_APPA:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< APPEARANCE ADAPTATION
                    s2t_img_raw = appa_net(s_imgs_raw)
                    s2t_imgs = self.may_augment_radiometric(s2t_img_raw, inplace=False)
                    s2t_imgs_for_dis = s2t_imgs

                    if DA.AUX_GEN:
                        z = torch.randn(cf.TRAIN.BTSZ, cf.AUX_GEN_MODEL.IN_CHN, 1, 1, device=self.device)
                        a_imgs_raw = appa_net.process(aux_gen(z))
                        aux_imgs = self.may_augment_radiometric(a_imgs_raw, inplace=False)
                        s2t_imgs_for_dis = torch.cat((s2t_imgs_for_dis, aux_imgs), dim=0)

                    L_gen = torch.mean(-torch.log(img_dis(s2t_imgs_for_dis) + 1e-9))  # --------- D(G(S)) -> 1 (T)
                    loss += L_gen * DA.W_GAN

                    if DA.APPA_SHIFT:  # ------------ Random shift of G(X) before passing it to S
                        rx, ry = np.random.randint(0, 8, 2)
                        rsx, rsy = (slice(rx, rx + cf.TRAIN.IN_SIZE)), (slice(ry, ry + cf.TRAIN.IN_SIZE))
                        sx, sy = (slice(4, cf.TRAIN.IN_SIZE - 4)), (slice(4, cf.TRAIN.IN_SIZE - 4))
                        pad_s2t = torch.nn.functional.pad(s2t_imgs, pad=(4, 4, 4, 4), mode='constant', value=0.0)
                        pad_lbls = torch.nn.functional.pad(s_lbls, pad=(4, 4, 4, 4), mode='constant', value=0)
                        crop_s2t = pad_s2t[:, :, rsx, rsy]
                        crop_lbls = pad_lbls[:, rsx, rsy]
                    else:
                        sx, sy = (slice(0, cf.TRAIN.IN_SIZE)), (slice(0, cf.TRAIN.IN_SIZE))
                        crop_s2t = s2t_imgs
                        crop_lbls = s_lbls

                    if DA.SHARED_FW:
                        tools.set_track_bn_running_stats(seg, DA.BATCH_NORM == 'MIX')
                        self.once(f"BN UPDATE ON MIX: {DA.BATCH_NORM == 'MIX'}")
                        s2t_preds, s_preds = tools.catted_forward(seg, (crop_s2t, s_imgs))
                    else:
                        tools.set_track_bn_running_stats(seg, DA.BATCH_NORM in ['SD', 'MIX'])
                        self.once(f"BN UPDATE ON SEG: {DA.BATCH_NORM in ['SD', 'MIX']}")
                        s_preds = seg(s_imgs)
                        tools.set_track_bn_running_stats(seg, DA.BATCH_NORM in ['S2T', 'MIX'])
                        self.once(f"BN UPDATE ON S2T: {DA.BATCH_NORM in ['S2T', 'MIX']}")
                        s2t_preds = seg(crop_s2t)

                    L_sup_s2t = self.criterion(s2t_preds[:, :, sx, sy], crop_lbls[:, sx, sy])
                    loss += L_sup_s2t * DA.W_TRA
                elif DA.REQUIRES_SD:
                    tools.set_track_bn_running_stats(seg, DA.BATCH_NORM in ['SD', 'MIX'])
                    self.once(f"BN UPDATE ON SD: {DA.BATCH_NORM in ['SD', 'MIX']}")
                    s_preds = seg(s_imgs)
                tools.set_track_bn_running_stats(seg, new_value=False)

                if DA.DO_REPA:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REPRESENTATION MATCHING
                    t_reps = seg.getReps(t_imgs)
                    rd_p_t = rep_dis(t_reps)
                    if type(rd_p_t) is not list:
                        rd_p_t = [rd_p_t, ]
                    else:
                        assert type(DA.W_REP) is list
                    if type(DA.W_REP) is not list:
                        DA.W_REP = [DA.W_REP, ]
                    for el in range(len(rd_p_t)):
                        rd_loss_t = torch.mean(-torch.log(rd_p_t[el] + 1e-9))  # ------------ D(R(T)) -> 1 (S)
                        loss += DA.W_REP[el] * rd_loss_t

                if self.e >= DA.SEG_PATIENCE and DA.W_SUP > 0.0:
                    L_sup_s = self.criterion(s_preds, s_lbls)
                    loss += L_sup_s * DA.W_SUP

                if DA.DO_ENTMIN:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< T-DOM ENTROPY MINIMIZATION
                    t_pr = torch.softmax(seg(t_imgs), dim=1)
                    entropies = -torch.sum(t_pr * torch.log(t_pr + 1e-5), dim=1, keepdim=True)
                    loss += torch.mean(entropies) * DA.W_ENT

                # ==================================================================================== BW-PASS / SGD
                seg_opt.zero_grad()
                if DA.DO_APPA:
                    appa_opt.zero_grad()
                    if DA.AUX_GEN: aux_gen_opt.zero_grad()
                if loss.item() > 0: loss.backward()
                if self.e >= DA.SEG_PATIENCE: seg_opt.step()
                if DA.DO_APPA:
                    appa_opt.step()
                    if DA.AUX_GEN: aux_gen_opt.step()

                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< UPDATE BN USING T-SAMPLES (adaBN)
                if DA.BATCH_NORM == 'TD':
                    tools.set_track_bn_running_stats(seg, new_value=True)
                    with torch.no_grad():
                        self.once("DOING LATE BN UPDATE ON: t_imgs_raw")
                        seg(t_imgs_raw)
                    tools.set_track_bn_running_stats(seg, new_value=False)

                # ============================================================================== TRAIN DISCRIMINATOR(S)
                tools.may_change_requires_grad(img_dis, DA.DO_APPA, True)
                tools.may_change_requires_grad(rep_dis, DA.DO_REPA, True)

                if DA.DO_APPA:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMG DIS UPDATE
                    s2t_imgs_for_dis = s2t_imgs.detach()
                    tdm_imgs_for_dis = t_imgs.detach()
                    if DA.AUX_GEN:
                        s2t_imgs_for_dis = torch.cat((s2t_imgs_for_dis[:DA.NON_AUX_BS], aux_imgs.detach()))

                    d_p_tdm = img_dis(tdm_imgs_for_dis)
                    d_p_s2t = img_dis(s2t_imgs_for_dis)

                    d_loss_tdm = torch.mean(-torch.log(d_p_tdm + 1e-9))  # ----------------- D(Xt) -> 1 (T)
                    d_loss_s2t = torch.mean(-torch.log(1 - d_p_s2t + 1e-9))  # ------------- D(A(Xs)) -> 0 (Non T)
                    d_std_tdm = d_p_tdm.std() if DA.DIS_REG else c0
                    d_std_s2t = d_p_s2t.std() if DA.DIS_REG else c0

                    dis_loss = (d_loss_tdm + d_loss_s2t + (d_std_tdm + d_std_s2t) * DA.DIS_REG) * DA.W_GAN
                    tools.update_network(dis_opt, dis_loss)

                if DA.DO_REPA:  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REP DIS UPDATE
                    with torch.no_grad():
                        t_reps = seg.getReps(t_imgs)
                        s_reps = seg.getReps(s_imgs)

                    rd_p_t = rep_dis(t_reps)
                    rd_p_s = rep_dis(s_reps)

                    if type(rd_p_t) is not list:
                        rd_p_t = [rd_p_t, ]
                        rd_p_s = [rd_p_s, ]

                    rep_dis_loss = torch.clone(c0)
                    for el in range(len(rd_p_t)):
                        rd_loss_t = torch.mean(-torch.log(1 - rd_p_t[el] + 1e-9))  # ---- D(R(Xt)) -> 0 (Non S)
                        rd_loss_s = torch.mean(-torch.log(rd_p_s[el] + 1e-9))  # -------- D(R(Xs)) -> 1 (S)
                        rep_dis_loss += (rd_loss_t + rd_loss_s)

                    tools.update_network(rep_dis_opt, rep_dis_loss)

            # ============================================================================== PRINTS, SAVES, ETC

            torch.cuda.empty_cache()
            print(f'  L_sup: {L_sup_s.item():.2f}', end='')
            if DA.DO_APPA: print(f'L_sup_S2T: {L_sup_s2t.item():.2f}, L_gen: {t2n(L_gen):.2f}', end='')
            if DA.DO_REPA: print(f'L_rep: {L_rep.item():.2f}, ', end='')
            print('')

            if DA.DO_APPA:
                d_acc_tdm = 100 * np.sum(t2n(d_p_tdm) > 0.5) / np.prod(t2n(d_p_tdm).shape)
                d_acc_s2t = 100 * np.sum(t2n(d_p_s2t) < 0.5) / np.prod(t2n(d_p_s2t).shape)
                print(f'  IMG_DIS Loss t: {t2n(d_loss_tdm):.2f} ({d_acc_tdm:.2f}%) STD: {d_std_tdm.item():.4f}', end='')
                print(f', s2t: {t2n(d_loss_s2t):.2f} ({d_acc_s2t:.2f}%) STD: {d_std_s2t.item():.4f} ')

            if cf.DA.DO_REPA:
                d_acc_tdm = 100 * np.sum(t2n(rd_p_t[-1]) < 0.5) / np.prod(t2n(rd_p_t[-1]).shape)
                dis_s_acc = 100 * np.sum(t2n(rd_p_s[-1]) > 0.5) / np.prod(t2n(rd_p_s[-1]).shape)
                print(f'  REP_DIS Loss t: {t2n(rd_loss_t):.2f} ({d_acc_tdm:.2f}%)', end='')
                print(f', s: {t2n(rd_loss_s):.2f} ({dis_s_acc:.2f}%)')

            self.may_save_model('latest.pt')
            self.all2eval()
            if DA.REQUIRES_SD:
                cmt = self.quick_eval_seg_on_training_batch()
                self.may_update_weights(cmt)
            self.may_save_target_dom_samples()
            if DA.DO_APPA:
                self.quick_eval_appa_net()
                if DA.AUX_GEN:
                    self.quick_eval_aux_gen()

            self.evaluate_segmentation_on_subset('target_dom_val')
            self.evaluate_segmentation_on_subset('target_dom_test')

    def eval_appa_network(self):
        """Apply appearance adaptation network to whole SD_TEST_SET in sliding window approach.

        Overlapping predictions are averaged. Uses de-normalization from cf.TD.
        """
        cf = self.cf
        appa_net = self.init_network(net_name='ap_ad', make_optimizer=False)
        self.restore_checkpoint(self.load_checkpoint())
        self.all2eval()

        self.prepare_datasets()
        images_ds, names_ds = self.tes_dataset.images, self.tes_dataset.names

        Ts, Cs = [], []  # --- Init Ts: pixel values , Cs: prediction counts
        for Ii in images_ds:
            hi, wi, _ = Ii.shape
            Ts.append(np.zeros((cf.AP_AD_MODEL.OUT_CHN, hi, wi), dtype=np.float32))
            Cs.append(np.zeros((1, hi, wi), dtype=np.float32))

        # -------------------------------------------------------------------------------- ITERATE OVER PATCHES
        in_size = cf.EVALUATION.IN_SIZE
        for batch in self.tes_data_loader:
            images, coordinates = batch['image'].to(self.device), batch['patch']
            upper_b = 5 if cf.EVALUATION.FLIP else 2
            for p in range(1, upper_b):
                with torch.no_grad():
                    images_st = t2n(may_permute(appa_net(may_permute(images, p)), p))
                for i, (pid, px, py,) in enumerate(coordinates):
                    Ts[pid][:, px:px + in_size, py:py + in_size] += images_st[i]
                    Cs[pid][0, px:px + in_size, py:py + in_size] += 1.0

        # ---------------------------------------------------------------------------------------- SAVE IMAGES
        for (Ii, Ti, Ci, Ni) in zip(images_ds, Ts, Cs, names_ds):
            img_dict = tools.denorm4print(Ii, domain=cf.SD, input_order='HWC')
            self.save_img_dict(img_dict, pjoin(self.F['adapted_images'], f'{Ni}-in'))
            img_dict = tools.denorm4print(Ti / Ci, domain=cf.TD, input_order='CHW')
            self.save_img_dict(img_dict, pjoin(self.F['adapted_images'], f'{Ni}-out'))

    # ============================================================================================ EVALUATION FUNCTIONS

    def evaluate_segmentation_on_subset(self, name: str):
        """Performs sliding-window based evaluation of the segmentation network.

        Writes confusion matrices and saves images for the subset.
        If cf.MODE != 'seg' and name == 'target_dom_val' the result is used for parameter selection / ES.
        Otherwise, the 'validation' set is used.
        :param name: The case to be evaluated in ['validation','testing','target_dom_val','target_dom_test']
        """
        cf = self.cf

        if name == 'validation':
            if not cf.EVALUATION.SD_VAL_SET: return
            dataset = self.val_dataset
            data_loader = self.val_data_loader
        elif name == 'testing':
            if not cf.EVALUATION.SD_TEST_SET: return
            dataset = self.tes_dataset
            data_loader = self.tes_data_loader
        elif name == 'target_dom_val':
            if not self.cf.TD or not cf.EVALUATION.TD_VAL_SET: return
            dataset = self.ada_val_dataset
            data_loader = self.ada_val_data_loader
        elif name == 'target_dom_test':
            if not self.cf.TD or not cf.EVALUATION.TD_TEST_SET: return
            dataset = self.ada_tes_dataset
            data_loader = self.ada_tes_data_loader
        else:
            raise NotImplementedError(f"Invalid case for segmentation evaluation: {name}")

        if dataset is None or not isinstance(dataset, EvalDataset):
            mprint(f'Evaluation of {name} is skipped', 2, cf.VRBS)
            return

        # ============================================================================ STEP 1: forward pass aggregation
        net = self.nets['seg']
        in_size = cf.EVALUATION.IN_SIZE
        Ps, Cs = [], []  # Aggregated probabilities / number of predictions

        for Ti in dataset.labels:
            hi, wi = Ti.shape
            Ps.append(np.zeros((cf.DATA.NCLS, hi, wi), dtype=np.float32))
            Cs.append(np.zeros((1, hi, wi), dtype=np.int16))

        for batch in data_loader:
            images, coordinates = batch['image'].to(self.device), batch['patch']
            for permutation in range(1, 5 if cf.EVALUATION.FLIP else 2):
                with torch.no_grad():
                    logits = may_permute(net(may_permute(images, permutation)), permutation)
                    probabilities = t2n(torch.softmax(logits, dim=1)).astype(np.float32)
                for i, (pid, px, py) in enumerate(coordinates):
                    Ps[pid][:, px:px + in_size, py:py + in_size] += probabilities[i]
                    Cs[pid][0, px:px + in_size, py:py + in_size] += 1

        # =============================================================================== STEP 2: averaging predictions
        self.CM *= 0
        Ls, Es = [], []  # Predicted label maps / entropy maps

        for i, (Pi, Ci, Ti, Ni) in enumerate(zip(Ps, Cs, dataset.labels, dataset.names)):
            Pi /= Ci
            Li = np.argmax(Pi, 0).astype(np.byte)
            Ls.append(Li)
            dwe.update_confusion_matrix(self.CM, Li, Ti.astype(int))
            if cf.EVALUATION.ENTROPY:
                entropies = -np.sum(Pi * np.log(Pi + 1e-5), axis=0, keepdims=True) / np.log(cf.DATA.NCLS)
                Es.append(entropies.astype(np.float16))

        # =============================================================================== STEP 3: store / print results
        cm_suffix = f"{name.upper()}_{self.e}" if cf.STAGE == 'train' else f"{name.upper()}"
        np.save(pjoin(self.F['confusion_matrices'], f'CM_{cm_suffix}'), self.CM)

        score = self.main_metric_from_cm()
        if (score > self.top_scores[name]) or self.e == 0:
            self.top_scores[name] = score
            if cf.MODE == 'source_training' and name == 'validation': self.e_es = 0
            if not (name in ['target_dom_val', 'target_dom_test', ]):
                self.may_save_model(f'{name}.pt')
        mprint(f'  {name.upper()} {cf.EVALUATION.METRIC} {score:.3f}/ {self.top_scores[name]:.3f}', 1, cf.VRBS)

        if name == 'target_dom_val' and cf.EVALUATION.ENTROPY and self.e >= cf.EVALUATION.MIN_ENT_ITER:
            mean_ent = np.mean(np.concatenate([np.ravel(e) for e in Es], axis=0))
            if mean_ent < self.top_scores['target_dom_val_E']:
                self.top_scores['target_dom_val_E'] = mean_ent
                if cf.MODE != 'source_training' and name == 'target_dom_val': self.e_es = 0
                self.may_save_model('TD_min_ent.pt')
            mprint(f'  {name.upper()} Entropy {mean_ent:.3f}/ {self.top_scores["target_dom_val_E"]:.3f}', 1, cf.VRBS)

        if ((name == 'validation' and cf.OUTPUTS.SAVE_VAL) or (name == 'testing' and cf.OUTPUTS.SAVE_TEST) or
                (name in ['target_dom_val', 'target_dom_test', ] and cf.OUTPUTS.SAVE_TDOM)):
            denorm_dom = cf.SD if name not in ['target_dom_val', 'target_dom_test', ] else cf.TD
            self.save_images(name, dataset.images, dataset.names, denorm_dom, Ls, dataset.labels)

    def quick_eval_seg_on_training_batch(self):
        """Estimate performance on training data.

        Function returns the confusion matrix, which may be used to update the class weights in ACE loss.
        :return: confusion matrix on training data from source domain
        """
        seg = self.nets['seg']
        self.CM *= 0
        for n_b in range(32):
            batch = next(iter(self.tra_data_loader))
            images, labels = batch['image'].to(self.device), batch['idmap']
            self.may_augment_radiometric(images, True)
            with torch.no_grad():
                logits = seg(images)
                preds = torch.argmax(logits, 1)
            dwe.update_confusion_matrix(self.CM, t2n(preds), t2n(labels))

        np.save(pjoin(self.F['confusion_matrices'], f'CM_TRAINING_{self.e}'), self.CM)
        score = self.main_metric_from_cm()

        self.top_scores['training'] = max(self.top_scores['training'], score)
        print(f'  TRAINING   SCORE/BEST {score:.1%} / {self.top_scores["training"]:.1%} ')

        if self.cf.OUTPUTS.SAVE_TRAIN: self.save_training_samples(images, preds, labels, self.cf.SD)
        return np.copy(self.CM)

    def quick_eval_appa_net(self, append_predictions=True):
        """ Save exemplary adapted images and optionally the corresponding references.

        A maximum number of 4 images per epoch is written.
        :param append_predictions: If True, the segmentation result of the adapted images will be attached and saved.
        """
        s_batch = next(iter(self.tra_data_loader))
        images, labels = s_batch['image'].to(self.device), s_batch['idmap']

        with torch.no_grad():
            s2t_img = self.nets['ap_ad'](images)
            if append_predictions:
                predictions = torch.argmax(self.nets['seg'](s2t_img), 1)

        for i in range(min(self.cf.TRAIN.BTSZ, 4)):
            img_dict = tools.denorm4print(images[i], self.cf.SD, input_order='CHW')
            for name, image in tools.denorm4print(s2t_img[i], self.cf.TD, input_order='CHW').items():
                img_dict[name + '-s2t'] = image
            if append_predictions: img_dict['pred'] = I2C(t2n(predictions[i][:, :]).astype(int), self.cf.SD)
            img_dict['ref'] = I2C(t2n(labels[i][:, :]).astype(int), self.cf.SD)
            self.save_img_dict(img_dict, pjoin(self.F['adapted_images'], f'ep{self.e}-{i}'))

    def quick_eval_aux_gen(self):
        """ Save exemplary images produced by the auxiliary generator.

        A maximum number of 4 images per epoch are saved.
        """
        aux_gen = self.nets['aux_gen']
        appa_net = self.nets['ap_ad']

        with torch.no_grad():
            z = torch.randn(self.cf.TRAIN.BTSZ, self.cf.AUX_GEN_MODEL.IN_CHN, 1, 1).to(self.device)
            aux_imgs = appa_net.process(aux_gen(z))

        for i in range(min(self.cf.TRAIN.BTSZ, 4)):
            img_dict = tools.denorm4print(t2n(aux_imgs[i]), self.cf.TD, input_order="CHW")
            self.save_img_dict(img_dict, pjoin(self.F['aux_generator'], f'ep{self.e}-{i}'))


# ======================================================================================= DATASET INFO

def describe(cf: config.Config):
    """ This function will compute and print statistics about a dataset.

    :param cf: Configuration object
    """
    class_stats = cf.STATS.CLASS_STATS
    channel_stats = cf.STATS.CHANNEL_STATS
    dom = cf.STATS.DOM
    subset = cf.STATS.SET

    if dom in dmng.nrw_codes:
        root = cf.PATHS.GeoNRW
        images, labels, _ = dmng.preload_nrw(root, dom, subset)
    else:
        raise NotImplementedError(f"Dataset {dom} not implemented in 'describe'")

    num_cha = cf.DATA.N_CHN
    num_sam = len(images)

    raveled_chs = [np.concatenate(tuple([images[i][:, :, c].ravel() for i in range(num_sam)]))
                   for c in range(num_cha)]
    raveled_lbs = np.concatenate(tuple([labels[i].ravel() for i in range(num_sam)]))
    npx = np.sum(raveled_lbs.shape[0])
    print(f"Dataset info:\n - # channels: {num_cha}\n - # samples: {num_sam}\n - # px: {npx}")

    ## CLASS STATS
    if class_stats:
        for ci in range(int(np.min(raveled_lbs)), int(1 + np.max(raveled_lbs))):
            print('Class {:d} ({:5.3f} % of data)'.format(ci, 100 * float(np.sum(raveled_lbs == ci)) / npx))
            print('         min    max   mean   stddev')
            where = raveled_lbs == ci
            for c in range(num_cha):
                sub = raveled_chs[c][where].astype(np.float)
                if len(sub) == 0:
                    print(" Ch{:2d}:{:7.2f}{:7.2f}{:7.2f}{:7.2f}".format(c, 0.0, 0.0, 0.0, 0.0))
                else:
                    print(" Ch{:2d}:{:7.2f}{:7.2f}{:7.2f}{:7.2f}".format(c, np.min(sub), np.max(sub), np.mean(sub),
                                                                         np.std(sub)))

    ## CHANNEL-WISE STATS
    if channel_stats:
        print(f'\nOverall')
        print('         min    max   mean   stddev')
        for c in range(num_cha):
            sub = raveled_chs[c].astype(float)
            print(" Ch{:2d}:{:7.2f}{:7.2f}{:7.2f}{:7.2f}".format(
                c, np.min(sub), np.max(sub), np.mean(sub), np.nanstd(sub)))

    ## CORRELATIONS
    for c in range(num_cha):
        print(f"Channel {c}\n - # mean: {np.mean(raveled_chs[c])}\n - # stddev: {np.std(raveled_chs[c])}")
        for q in range(c + 1, num_cha):
            print(f" - correl. with {q}: {np.corrcoef(raveled_chs[c], raveled_chs[q])[0][1]}")

    print('=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:\n\n')


# ================================================================================================================ MAIN

def run_experiment(cf_path):
    c = config.Config(cf_path)
    c.check_consistency(config.Config('./Documentation/_config_documentation.yaml'))

    if c.MODE == 'source_training':
        H = ExperimentHandler(c)
        if c.STAGE == 'train':
            H.train_seg()
        elif c.STAGE == 'eval':
            H.eval_seg()
        del H

    elif c.MODE == 'domain_adaptation':
        H = ExperimentHandler(c)
        if c.STAGE == 'train':
            H.train_da()
        elif c.STAGE == 'eval':
            H.eval_seg()
        elif c.STAGE == 'eval-appa':
            H.eval_appa_network()
        del H

    elif c.MODE == 'dataset':
        if c.STAGE == 'describe':
            describe(c)

    del c
    print('Experiment done!\n')


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python main.py path/to/config.yaml or use experiment_scheduler.py"
    run_experiment(sys.argv[1])
