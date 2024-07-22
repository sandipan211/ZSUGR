import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):

    def __init__(self, res_size,att_size,layer_sizes):
        super().__init__()
        self.MLP = nn.Sequential()
        layer_sizes+=[1]
        for i, (in_size, out_size) in enumerate(zip([res_size+att_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)

    def forward(self, x,s):
        h = torch.cat((x, s), dim=-1)
        h = self.MLP(h)
        return h

class Generator(nn.Module):

    def __init__(self, layer_sizes, latent_size, attSize):
        super().__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size + attSize

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
            else:
                self.MLP.add_module(name="ReLU", module=nn.ReLU(inplace=True))
        self.apply(weights_init)

    def forward(self, z, s):
        h = torch.cat((z, s), dim=-1)
        x = self.MLP(h)
        return x

class Mapping_net(nn.Module):

    def __init__(self, layer_sizes, att_size):
        super().__init__()
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([att_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A%i" % (i), module=nn.LeakyReLU(0.2, True))
        self.apply(weights_init)

    def forward(self, s):
        w = self.MLP(s)
        return w