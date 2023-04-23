import os, sys
from main import run_experiment


# Note: run this script from ./code/: python experiment_scheduler.py ./rel/path/to/root/


def run_folder(root):
    """ Runs experiment according to all configuration files in a root folder.

    :param root: path to a folder that contains multiple configuration files
    """
    yfiles = [f for f in os.listdir(root) if f.endswith('.yaml')]
    yfiles.sort()
    print('Scheduler is iterating over folder', root)
    for i, file in enumerate(yfiles):
        print(f'Schedular starts file {i}: {file}')
        run_experiment(os.path.join(root, file))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(sys.argv)
        print("Please provide folder(s)")
        print("->  python experiment_scheduler.py ./rel/path/to/root/ ...")
        exit()

    for folder in sys.argv[1:]:
        run_folder(folder)
