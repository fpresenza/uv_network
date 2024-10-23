import csv
import numpy as np


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