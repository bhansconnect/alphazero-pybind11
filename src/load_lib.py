import glob
import importlib.util
import os


def load_alphazero():
    src_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(os.path.dirname(src_path), 'build/src')
    lib_path = glob.glob(os.path.join(build_path, 'alphazero*.so'))[0]

    spec = importlib.util.spec_from_file_location(
        'alphazero', lib_path)
    alphazero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alphazero)
    return alphazero
