import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model

epoch = int(sys.argv[1])
#TODO: Enter your model here
model=model.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
