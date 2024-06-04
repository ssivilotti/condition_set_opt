import numpy as np

def convert_to_onehot(shape: tuple, point:list)-> int:
    onehot = np.zeros(np.sum(shape))
    for i, p in enumerate(point):
        onehot[int(p + np.sum(shape[:i]))] = 1
    return onehot

def convert_from_onehot(shape: tuple, onehot: int)-> list:
    point = []
    num = 0
    shape_counter = 0
    for i in range(len(onehot)):
        if onehot[i] == 1:
            point.append(num)
        num += 1
        if num >= shape[shape_counter]:
            num = 0
            shape_counter += 1
    return point