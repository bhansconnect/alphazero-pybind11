import glob
import importlib.util
import os


def load_alphazero():
    src_path = os.path.dirname(os.path.realpath(__file__))
    build_path = os.path.join(os.path.dirname(src_path), "build/src")
    so_path = glob.glob(os.path.join(build_path, "alphazero*.so"))
    pyd_path = glob.glob(os.path.join(build_path, "alphazero*.pyd"))
    lib_path = (pyd_path + so_path)[0]

    spec = importlib.util.spec_from_file_location("alphazero", lib_path)
    alphazero = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alphazero)
    return alphazero
