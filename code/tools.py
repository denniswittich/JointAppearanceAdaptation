import os, datetime
import numpy as np
import torch

nrw_codes = ['aachen', 'bochum', 'dortmund', 'heinsberg', 'muenster', ]  # You may add any other city from GeoNRW

# ======================================================================================================== COLOR CODES

colors_nrw = np.array(  # ---------- Original color codes (unused!)
    [[0, 0, 0],  # -------- 0: Unknown
     [44, 160, 44],  # ---- 1: Forest
     [31, 119, 180],  # --- 2: Water ---------------> Unknown
     [140, 86, 75],  # ---- 3: Agricultural
     [127, 127, 127],  # -- 4: Urban
     [188, 189, 34],  # --- 5: Grassland
     [255, 127, 14],  # --- 6: Railway -------------> Unknown
     [149, 103, 189],  # -- 7: Highway
     [23, 190, 207],  # --- 8: Airports_shipyards --> Unknown
     [214, 39, 40],  # ---- 9: Road
     [227, 119, 194],  # - 10: Building
     ])

colors_nrw_mapped = np.array(  # ---------- classes to predict: 7
    [[227, 119, 194],  # -- 0: Building
     [44, 160, 44],  # ---- 1: Forest
     [214, 39, 40],  # ---- 2: Road
     [140, 86, 75],  # ---- 3: Agricultural
     [127, 127, 127],  # -- 4: Urban
     [0, 0, 0],  # -------- 6: Unknown
     ])



def idmap2color(idmap: np.ndarray, dom: str) -> np.ndarray:
    """Map id maps to colored label images"""
    h, w = idmap.shape[:2]
    if dom in nrw_codes:
        colour_map = colors_nrw_mapped[idmap.reshape(-1)].reshape((h, w, 3))
    else:
        raise NotImplementedError(f"Dataset {dom} not implemented in 'idmap2color'")
    return colour_map.astype(np.ubyte)


def color2idmap(colour_map: np.ndarray, dom: str) -> np.ndarray:
    """Map colored label images to id maps"""
    if dom in nrw_codes:
        colour_map = np.repeat(colour_map[:, :, np.newaxis, :], 7, axis=-2)
        match = np.all(colour_map == colors_nrw_mapped, axis=-1)
    else:
        raise NotImplementedError(f"Dataset {dom} not implemented in 'color2idmap'")

    # color_ok = np.sum(match, axis=-1) == 1  # << OPTIONAL: Check if there are no other colors
    # if not np.all(color_ok): print(colour_map[np.where(np.logical_not(color_ok))])

    idmap = np.argmax(match, axis=-1).astype(np.ubyte)
    return idmap

# ======================================================================================== Denormalization for printing


def denorm4print(input_image, domain: str, input_order: str = 'HWC') -> dict:
    """Converts tensor/array to de-normalized visualizations with names.

    Note that the dict format allows to return multiple visualizations per sample.
    :param input_image: Input image (torch.tensor or numpy.ndarray)
    :param domain: Domain code (affects de-normalization)
    :param input_order: HWC channel order as string (e.g. 'HWC' or 'CHW')
    :return: dict with names and images ready to be displayed/stored
    """

    assert len(input_order) == 3, "Function 'denorm4print' supports only single images"

    if not isinstance(input_image, np.ndarray):
        input_image = tensor2numpy(input_image)

    if input_order != 'HWC':  # ------------------------------------------------- REORDER CHANNELS TO HWC
        order = (input_order.find('H'), input_order.find('W'), input_order.find('C'))
        input_image = input_image.transpose(order)

    if domain in nrw_codes:
        img = np.clip((input_image + 1) * 127.5, 0, 255).astype(np.ubyte)
    else:
        raise NotImplementedError(f"Dataset {domain} not implemented in 'denorm4print'")

    return {'img': img, }


# =============================================================================================== NETWORK MODIFICATIONS

def update_network(network_optimiser, loss):
    """Do one step with optimiser to minimise a loss"""
    network_optimiser.zero_grad()
    loss.backward()
    network_optimiser.step()


def set_track_bn_running_stats(network, new_value: bool):
    """Enable or disable update of running averages"""
    if new_value:
        network.apply(enable_bn_running_stats)
    else:
        network.apply(disable_bn_running_stats)


def enable_bn_running_stats(m):
    """Used in set_track_bn_running_stats"""
    class_name = m.__class__.__name__
    if class_name.find("BatchNorm") != -1:
        m.track_running_stats = True


def disable_bn_running_stats(m):
    """Used in set_track_bn_running_stats"""
    class_name = m.__class__.__name__
    if class_name.find("BatchNorm") != -1:
        m.track_running_stats = False


def may_change_requires_grad(network, condition: bool, new_value: bool) -> None:
    """Change the value of net.requires_grad to 'new value' if 'condition' is fulfilled.

    :param network: Network
    :param condition: If this condition is fulfilled, the value of 'requires_grad' is set to 'new_value'
    :param new_value: Boolean value to which 'requires_grad' is set to
    """
    if condition:
        for param in network.parameters():
            param.requires_grad = new_value


# =============================================================================================================== MIXED


class FolderDict:
    data: dict

    def __init__(self, data):
        """This class stores a dict where the values are paths.

        The path is only created if the item is accessed by __getitem__.
        :param data: The dictionary containing folder paths.
        """
        self.data = data

    def __getitem__(self, item):
        path = self.data[item]
        os.makedirs(path, exist_ok=True)
        return path


class Once:
    printed: list

    def __init__(self):
        """This object can be used to make sure messages are printed only once.

        In particular, a message is printed only on the first __call__.
        """
        self.printed = []

    def __call__(self, text):
        if text in self.printed:
            return
        else:
            print(text)
            self.printed.append(text)


def may_overlay_grid(img_dict: dict, grid_size: int) -> None:
    """Superimpose a black grid on each image in the dict."""
    if grid_size > 0:
        for _, img in img_dict.items():
            h, w = img.shape[:2]
            for d in (0, -1):
                img[d, :] *= 0
            for d in (0, -1):
                img[:, d] *= 0
            for d in range(0, h, grid_size):
                img[d, :, :] //= 2
            for d in range(0, w, grid_size):
                img[:, d, :] //= 2


def tensor2numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a np.ndarray"""
    return t.cpu().data.numpy()


def catted_forward(network, tensors:list) -> list:
    """Jointly process a list of tensors by a network.

    Tensors are concatenated along the batch dimension.
    :param network: The network to process the tensors.
    :param tensors: A list or tuple of tensors to jointly process.
    :return: A list of outputs corresponding to the tensors."""
    lengths = [t.shape[0] for t in tensors]
    out = network(torch.cat(tensors, dim=0))
    return torch.split(out, lengths, dim=0)


def may_print(text, min_verbosity:int, verbosity:int, no_line_break=False):
    """May print a text if verbosity >= min_verbosity"""
    if verbosity >= min_verbosity: print(text, end='' if no_line_break else '\n')


def current_datetime_as_str() -> str:
    """Return a string that represents the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def may_permute(tensor, case: int) -> torch.Tensor:
    """Permutation is used for making redundant predictions in the evaluation phase.

    The function will flip the tensor along horizontal, vertical or both axis.
    The permutation  is undone by applying the same permutation again.
    Non-tensors are returned without modification. Tensors need a (B,C,H,W) format.
    :param tensor: The tensor to permute.
    :param case: The type of permutation, indicated by an integer 1-4 (1 eans no permutation).
    """

    if isinstance(tensor, torch.Tensor):
        assert tensor.ndim == 4, "This function only accepts 4D tensors"
        if case == 2:
            return torch.flip(tensor, (2,))
        elif case == 3:
            return torch.flip(tensor, (3,))
        elif case == 4:
            return torch.flip(tensor, (2, 3))
        else:
            return tensor
    else:
        return tensor


def sek2hms(seconds: int) -> tuple:
    """Convert seconds to hour, minutes, seconds.

    Returns a tuple of three integers.
    :param seconds: The number of seconds
    """

    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60
    return hours, minutes, seconds
