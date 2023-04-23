import os

from yaml import load, Loader, dump

def joined_yaml(base: dict, new: dict):
    """ Update this config with base yaml file.
    Shared arguments will be overwritten by self

    :param base: base yaml object to be joined
    :param new: new yaml object to be joined
    :return raw joined text
    """

    joined = {}

    for key, v in base.items():
        if key == 'BASE': continue
        if isinstance(v, dict):
            if key in new.keys():  # ----------------------------------------------------------------------- update node
                joined[key] = joined_yaml(v, new[key])
            else:  # ------------------------------------------------------------------------------------ keep base node
                joined[key] = v.copy()
        else:
            if key in new.keys():
                joined[key] = new[key]
            else:
                joined[key] = v

    for key, v in new.items():
        if key == 'BASE': continue
        if key in base.keys(): continue
        if isinstance(v, dict):
            joined[key] = v.copy()
        else:
            joined[key] = v

    return joined


class Config(object):
    # NOTE: The inital attributes are only required for type hints in the IDE!
    # The helper function 'make_prototype_from_yaml' can be used to generate the respective code.
    # START OF AUTOMATED CODE
    class PROT_PATHS(object):
        GeoNRW: str

    PATHS = PROT_PATHS()

    MODE: str
    STAGE: str
    SD: str
    TD: str
    CUDA: str
    RANDOM_SEED: int
    VRBS: int

    class PROT_DATA(object):
        N_CHN: int
        NCLS: int
        IGNORE_INDEX: int

    DATA = PROT_DATA()

    class PROT_AUG(object):
        PRECROP: bool
        ROTATE: bool
        FLIP: bool
        RAND_RESCALE: int
        RAND_RESCALE: int
        RESCALE_SYM: bool
        RADIO_SCALE: float
        INTERPOLATE: bool

    AUG = PROT_AUG()

    class PROT_SEG_MODEL(object):
        IN_CHN: int
        OUT_CHN: int
        TYPE: str

        class PROT_UNET(object):
            DEPTH: int
            BACKBONE: str
            PRETRAINED: bool

        UNET = PROT_UNET()

    SEG_MODEL = PROT_SEG_MODEL()

    class PROT_AP_AD_MODEL(object):
        IN_CHN: int
        OUT_CHN: int
        TYPE: str

        class PROT_RNET(object):
            NUM_BLOCKS: int
            NUM_FEAT: int
            SCALE: int
            DROPRATE: float

        RNET = PROT_RNET()

    AP_AD_MODEL = PROT_AP_AD_MODEL()

    class PROT_AP_DIS_MODEL(object):
        IN_CHN: int
        OUT_CHN: int
        SHIFT: bool
        TYPE: str

    AP_DIS_MODEL = PROT_AP_DIS_MODEL()

    class PROT_AUX_GEN_MODEL(object):
        IN_CHN: int
        OUT_CHN: int
        TYPE: str
        DROPRATE: float
        START_SIZE: int

    AUX_GEN_MODEL = PROT_AUX_GEN_MODEL()

    class PROT_RP_DIS_MODEL(object):
        IN_CHN: int
        OUT_CHN: int
        TYPE: str
        NUM_F_START: int

    RP_DIS_MODEL = PROT_RP_DIS_MODEL()

    class PROT_TRAIN(object):
        SD_SET: str
        TD_SET: str
        IT_P_EP: int
        N_EP_MAX: int
        EARLY_STOPPING_EP: int
        BTSZ: int
        IN_SIZE: int
        NUM_WK: int
        PREFETCH_FACTOR: int

        class PROT_LOSS(object):
            TYPE: str
            ACE_POW: float
            FCL_GAMMA: float
            WEIGHTING: str

        LOSS = PROT_LOSS()

        class PROT_SEG(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: str
            LR_DEC: float

        SEG = PROT_SEG()

        class PROT_AP_AD(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: float
            LR_DEC: float

        AP_AD = PROT_AP_AD()

        class PROT_AP_DIS(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: float
            LR_DEC: float

        AP_DIS = PROT_AP_DIS()

        class PROT_AUX_GEN(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: float
            LR_DEC: float

        AUX_GEN = PROT_AUX_GEN()

        class PROT_RP_DIS(object):
            OPTIM: str
            LR: float
            BETA1: float
            BETA2: float
            WDEC: float
            LR_DEC: float

        RP_DIS = PROT_RP_DIS()

    TRAIN = PROT_TRAIN()

    class PROT_DA(object):
        JOINT: bool
        DO_APPA: bool
        DO_REPA: bool
        DO_ENTMIN: bool
        AUX_GEN: bool
        NON_AUX_BS: int
        REQUIRES_SD: bool
        SHARED_FW: bool
        BATCH_NORM: str
        REP_LAYER: str
        REP_ENCODER_STAGE: int
        SEG_PATIENCE: int
        APPA_WARMUP: int
        APPA_SHIFT: bool
        DIS_REG: float
        W_SUP: float
        W_TRA: float
        W_GAN: float
        W_REP: float
        W_ENT: float

    DA = PROT_DA()

    class PROT_EVALUATION(object):
        METRIC: str
        SD_VAL_SET: str
        SD_TEST_SET: str
        TD_VAL_SET: str
        TD_TEST_SET: str
        IN_SIZE: int
        SW_SHIFT: int
        FLIP: bool
        ENTROPY: bool
        MIN_ENT_ITER: int

    EVALUATION = PROT_EVALUATION()

    class PROT_CHECKPOINTS(object):
        LOAD_INIT: str
        LOAD_FROM: str
        LOAD_STRICT: bool
        SAVE: bool
        LOAD: bool
        LOAD_OPT: bool

    CHECKPOINTS = PROT_CHECKPOINTS()

    class PROT_OUTPUTS(object):
        GRID: int
        IMAGE_EXT: str
        SAVE_SINGLE: bool
        SAVE_TRAIN: bool
        SAVE_TDOM_TRAIN: bool
        SAVE_VAL: bool
        SAVE_TEST: bool
        SAVE_TDOM: bool
        FOLDER: str

    OUTPUTS = PROT_OUTPUTS()

    class PROT_STATS(object):
        DOM: str
        SET: str
        CLASS_STATS: bool
        CHANNEL_STATS: bool

    STATS = PROT_STATS()

    VERSION: float

    # END OF AUTOMATED CODE

    def __init__(self, path=None, yaml=None, root=""):
        """ Generate config object.
        Either path or yaml dict has to be provided.
        All configurations can be accessed in object style.
        BASE config will only be considered if the path is given.

        :param path: path to a config file
        :param yaml: yaml data as dict
        :param root: root of the config (will be automatically identified if path is given and used to replace ~CONFIG)
        """

        assert (path is None) ^ (yaml is None), "Either Path or YAML dict has to be provided"
        if not path is None:
            with open(path, 'r') as file:
                self.absfile = os.path.abspath(path)
                self.root = '/'.join(self.absfile.split(os.sep)[:-1])
                raw = file.read()
                raw = raw.replace("~CONFIG", self.root)
                raw = raw.replace("\t", "    ")
                self.yaml = load(raw, Loader)
        else:
            self.yaml = yaml
            self.root = root

        if 'BASE' in self.yaml.keys() and not path is None:
            # print(self.yaml['BASE'])
            base = Config(path=self.yaml['BASE'])
            self.yaml = joined_yaml(base.yaml, self.yaml)

        self.__set_attributes_from_yaml__()

    def __set_attributes_from_yaml__(self):
        """Transform yaml to attributes """

        for key, v in self.yaml.items():
            if key == 'BASE': continue
            if isinstance(v, (list, tuple)):
                new_v = [Config(yaml=x, root=self.root) if isinstance(x, dict) else x for x in v]
            else:
                new_v = Config(yaml=v, root=self.root) if isinstance(v, dict) else v
            setattr(self, key, new_v)

    def __str__(self):
        return dump(self.yaml)

    def __repr__(self):
        return dump(self.yaml)

    def check_consistency(self, prot: object):
        """Will throw AssertionError if a declared attribute is missing in prototype!

        :param prot: Prototype config object to compare this instance to.

        Note that no type-checks are done as some variables can be given as simple type or list!
        Not all arguments defined in the prototype have to be provided! (May lead to runtime errors)
        Exemplary usage:
         C = Config('../exp/tests/test_config.yaml')
         PROT = Config('./Documentation/_config_documentation.yaml')
         C.check_consistency(PROT)
        """
        for att in dir(self):
            if att.startswith('__'): continue
            if not att in dir(prot):
                print(f"WARNING: Node {att} is not listed in prototype!")
                continue
            att_v = self.__getattribute__(att)
            if issubclass(type(att_v), Config):
                att_v.check_consistency(prot.__getattribute__(att))


def make_prototype_from_yaml(path):
    """Creates the attribute code for using the config in the ide

    .
    """

    def append_node(node, indentation):
        code = ""
        for key, v in node.items():
            if isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, dict):
                        code += " " * indentation + f"class PROT_{key}(object):\n"
                        code += append_node(x, indentation + 4)
                        code += " " * indentation + f"{key} = PROT_{key}()\n\n"
                    else:
                        code += " " * indentation + f"{key}: {type(x).__name__}\n"
            else:
                if isinstance(v, dict):
                    code += " " * indentation + f"class PROT_{key}(object):\n"
                    code += append_node(v, indentation + 4)
                    code += " " * indentation + f"{key} = PROT_{key}()\n\n"
                else:
                    code += " " * indentation + f"{key}: {type(v).__name__}\n"
        return code

    with open(path, 'r') as file:
        raw = file.read()
        raw = raw.replace("\t", "    ")
        yaml = load(raw, Loader)

    code = append_node(yaml, 0)
    print(code)


if __name__ == '__main__':
    # If the exemplary yaml file for documentation was changed, run this file to update the code in the Config class
    make_prototype_from_yaml('./Documentation/_config_documentation.yaml')


