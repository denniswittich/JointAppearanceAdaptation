import torch
from torch import nn
import numpy as np
from numba import jit


class SoftmaxMaeLoss:
    def __init__(self, device: str, n_cls: int, ignore_index=None):
        """Softmax Mean Absolute Error loss.

        This loss computes the mean absolute distance between the reference probabilities and the predicted ones after softmax.
        The loss is called with arguments: predicted logits and reference labels as index map.
        :param device: Device on which auxiliary tensors should be created
        :param n_cls: Number of classes
        :param ignore_index: ID of the class to be ignored (or None if no class should be ignored)
        """

        self.n_cls = n_cls
        self.ignore_index = ignore_index
        self.scalar_float_zero = torch.zeros((1,), dtype=torch.float, device=device)
        self.scalar_int_zero = torch.zeros((1,), dtype=torch.int64, device=device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_flat = labels.view(-1, 1)  # ----------------- [NHW ,1]

        if self.ignore_index is not None:
            keeps = torch.logical_not(torch.eq(labels_flat.detach(), self.ignore_index))
            labels_flat_corr = torch.where(keeps, labels_flat, self.scalar_int_zero)
            logits_t = logits.transpose(1, 2).transpose(2, 3)
            logits_flat = logits_t.reshape(-1, self.n_cls)  # -------- [NHW , ncls]
            labels_oh = torch.zeros_like(logits_flat)  # construct one-hot encoded labels
            labels_oh.scatter_(1, labels_flat_corr, 1.0)

            softmax = torch.softmax(logits_flat, 1)  # N x C
            reduced_abs_diff = torch.mean(torch.abs(labels_oh - softmax), dim=1)
            kept_diffs = torch.where(keeps.view(-1), reduced_abs_diff, self.scalar_float_zero)

            loss = torch.sum(kept_diffs) / torch.sum(keeps.float())
            return loss

        else:
            logits_t = logits.transpose(1, 2).transpose(2, 3)
            logits_flat = logits_t.reshape(-1, self.n_cls)  # -------- [NHW , ncls]
            labels_oh = torch.zeros_like(logits_flat)  # construct one-hot encoded labels
            labels_oh.scatter_(1, labels_flat, 1.0)

            softmax = torch.softmax(logits_flat, 1)  # N x C
            loss = torch.mean(torch.abs(labels_oh - softmax))
            return loss


class DiceLoss:

    def __init__(self, n_cls: int, ignore_index=None):
        """Dice loss.

        Old implementation.. use with care!
        :param n_cls: Number of classes
        :param ignore_index: ID of the class to be ignored (or None if no class should be ignored)
        """
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.ignore_index = ignore_index
        self.n_cls = n_cls

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        # compute softmax over the classes axis, transpose to channels last
        labels_flat_ = labels.view(-1, 1)  # ----------------------- [NHW ,1]
        logits_t = torch.permute(logits, (0, 2, 3, 1))  # ------- channels last
        logits_flat_ = logits_t.reshape(-1, self.n_cls)  # --------- [NHW , ncls]

        if not self.ignore_index is None:
            do_use = labels_flat_[:, 0] != self.ignore_index
            labels_flat = labels_flat_[do_use]
            logits_flat = logits_flat_[do_use]
        else:
            labels_flat = labels_flat_
            logits_flat = logits_flat_

        labels_oh = torch.zeros_like(logits_flat)  # construct one-hot encoded labels
        labels_oh.scatter_(1, labels_flat, 1.0)

        softmax = torch.softmax(logits_flat, 1)  # N x C

        dims = (1,)
        intersection = torch.sum(softmax * labels_oh, dims)
        cardinality = torch.sum(softmax + labels_oh, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


@jit(nopython=True)
def sample_labels(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray):
    """Creates sampled labels based on predictions, labels and sampling probabilities.

    Any wrong classifications are kept! TODO consider borders ?!
    :param labels: The reference labels as id maps
    :param predictions: The predicted probability maps
    :param probabilities: Array containing the probability for each class to set the ref. not to -1
    """

    n, h, w = labels.shape
    sampled_labels = np.copy(labels)

    for x in range(h):
        for y in range(w):
            for i in range(n):
                p = predictions[i, x, y]
                l = labels[i, x, y]
                if l != p: continue

                if np.random.random() > probabilities[l]:
                    sampled_labels[i, x, y] = -1  # WARNING: hard-coded magic number, this should be the ignore index

    return sampled_labels


class SampledCrossEntropyLoss:
    def __init__(self, n_cls, ignore_index=-1):
        """Initialize sampled ce loss. Pixels are selected according to occurrence of each class.

        Old implementation.. use with care!
        NOTE: the function assumes that the labels are in [ignore_index, 0, ..., n_cls-1]
        The loss is called with arguments: predicted logits and reference labels as index map.
        :param n_cls: Number of classes
        :param ignore_index: ID of the class to be ignored (or None if no class should be ignored)
        """

        self.ignore_index = ignore_index
        self.n_cls = n_cls
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, logits, labels, return_sampled_labels=False):
        labels_np = labels.cpu().data.numpy()

        preds = torch.argmax(logits, 1)
        preds_np = preds.cpu().data.numpy()

        uniques, counts = np.unique(labels_np, return_counts=True)
        L = list(zip(uniques, counts))
        L.sort()
        uniques, counts = list(zip(*L))
        if uniques[0] == -1:
            uniques = uniques[1:]
            counts = counts[1:]

        nuq = len(uniques)

        if (nuq < self.n_cls):
            # print("Not all classes represented in batch!")
            probas = np.zeros(self.n_cls)
        else:
            occurrences = (counts / np.sum(counts)) + 1
            probas = np.min(occurrences) / occurrences

        # probas e.g. [0.74498568 0.83445653 0.84263872 0.87521844 1.        ]

        sampled_labels = sample_labels(labels_np, preds_np, probas)
        sparse_labels = torch.from_numpy(sampled_labels).to(labels.device)

        if return_sampled_labels:
            return self.ce_loss(logits, sparse_labels), sampled_labels
        else:
            return self.ce_loss(logits, sparse_labels)


class FocalCrossEntropyLoss:
    def __init__(self, n_cls, gamma=1.0, ignore_index=-1):
        """Compute focal loss according to the prob of the sample.

        Old implementation.. use with care!
        loss= -(1-p)^gamma*log(p)
        The loss is called with arguments: predicted logits and reference labels as index map.
        :param n_cls: Number of classes
        :param gamma: Weighting factor
        :param ignore_index: ID of the class to be ignored (or None if no class should be ignored)
        """

        self.ignore_index = ignore_index
        self.n_cls = n_cls
        self.gamma = gamma

    def __call__(self, logits, labels):
        labels_flat_ = labels.view(-1, 1)  # ----------------------- [NHW ,1]
        logits_t = torch.permute(logits, (0, 2, 3, 1))  # ------ channels last
        logits_flat_ = logits_t.reshape(-1, self.n_cls)  # --------- [NHW , ncls]

        if self.ignore_index is not None:
            do_use = labels_flat_[:, 0] != self.ignore_index
            labels_flat = labels_flat_[do_use]
            logits_flat = logits_flat_[do_use]
        else:
            labels_flat = labels_flat_
            logits_flat = logits_flat_

        labels_oh = torch.zeros_like(logits_flat)  # construct one-hot encoded labels
        labels_oh.scatter_(1, labels_flat, 1.0)

        eps = 1e-12
        softmax = torch.softmax(logits_flat, 1)  # N x C

        inv_softmax = torch.ones_like(softmax) - softmax
        inv_softmax = inv_softmax.detach()
        if self.gamma != 1.0:
            inv_softmax = torch.pow(inv_softmax, self.gamma)

        loss = -torch.mean(labels_oh * torch.log(softmax + eps) * inv_softmax)
        return loss

