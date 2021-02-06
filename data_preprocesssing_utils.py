import os

def check_dir(path):
  if path is None:
    return
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=False)