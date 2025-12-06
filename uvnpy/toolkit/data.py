import csv
import numpy as np
from itertools import islice


def write_csv(file_path, data, one_row=False):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        if one_row:
            writer.writerow(data)
        else:
            for row in data:
                writer.writerow(row)


def read_csv(
        file_path,
        rows=(0, 1),
        jump=1,
        dtype=object,
        shape=-1,
        asarray=False
        ):
    data = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)

        k = 0
        for i, row in enumerate(reader):
            if (i % jump == 0):
                if i >= rows[0] and i < rows[1]:
                    data.append(np.array(row).astype(dtype).reshape(shape))
                    k += 1
                elif i >= rows[1]:
                    break

    if asarray:
        return np.asarray(data)
    else:
        return data


def read_csv_numpy(file_path, start=None, stop=None, jump=1):
    with open(file_path, mode='r') as file:
        arr = np.loadtxt(islice(file, start, stop, jump), delimiter=",")

    return arr


def read_rows_csv(
        file_path,
        rows,
        dtype=object,
        shape=-1,
        asarray=False
        ):
    data = []

    with open(file_path, mode='r') as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            if i in rows:
                data.append(np.array(row).astype(dtype).reshape(shape))
            elif i > np.max(rows):
                break

    if asarray:
        return np.asarray(data)
    else:
        return data
