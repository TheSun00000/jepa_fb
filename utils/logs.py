import os
import shutil
import csv


def auto_cast(x):
    for cast in (int, float):
        try:
            return cast(x)
        except:
            pass
    return x

def parse_kwargs(argv):
    out = {}
    for item in argv:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k] = auto_cast(v)
    return out


def add_text_to_file(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        
        
        
# CSV ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

def init_csv_logger(path, fieldnames):
    file_exists = os.path.isfile(path)

    if file_exists:
        return  # don't overwrite existing file

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def log_csv_row(path, row_dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        writer.writerow(row_dict)