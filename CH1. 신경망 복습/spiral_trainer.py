# 1.4.4 Trainer 클래스
from pickletools import optimize
import sys
sys.path.append('..')
from backward_propagation import SGD
from common.trainer import Trainer
from dataset import spiral
from spiral_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()