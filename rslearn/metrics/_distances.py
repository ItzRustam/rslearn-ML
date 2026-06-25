# rslearn-ML
# Copyright (C) 2026 Rustam Singh Bhadouriya
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the LICENSE file for more details.

import numpy as np

def EuclidienDisctance(dataset : np.array = None, key : np.array = None):
    
    if dataset is None:
        raise ValueError("dataset is not given.")
    if key is None:
        raise ValueError("No Key given to Found.")
    
    dataset = np.asarray(dataset)
    key = np.asarray(key)

    
    if dataset.shape[1] != key.shape[0]:
        raise ValueError(f"Shape Mismatch {(dataset.shape, key.shape)}")
    
    
    distances = np.linalg.norm(dataset - key, axis=1)
    return distances

if __name__ == '__main__':
    data = np.array([
        [1,2],
        [3, 4],
        [4, 5]
    ])

    key = np.array([2, 3])
    dis = EuclidienDisctance(data, key)
    print(dis)
    