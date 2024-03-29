#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2022 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0

import random
from nn import Nn

a = Nn()
a.version()
a.add_layer(256, Nn.ActivationFunctionType.NONE.value, 0);
a.add_layer(40, Nn.ActivationFunctionType.LEAKY_RELU.value, 0);
a.add_layer(40, Nn.ActivationFunctionType.LEAKY_RELU.value, 0);
a.add_layer(10, Nn.ActivationFunctionType.SIGMOID.value, 0);

data = a.data_load('train.csv', a.model.contents.width[0], a.model.contents.width[a.model.contents.depth-1])
for j in range(1000):
	data = random.sample(data, len(data))
	for line in data:
		a.train(line[:a.model.contents.width[0]], line[a.model.contents.width[0]:], 0.01)

a.save('model.txt')

