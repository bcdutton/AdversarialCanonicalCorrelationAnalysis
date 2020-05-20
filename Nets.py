import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init

# https://github.com/1Konny/Beta-VAE/blob/master/model.py
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def encoder(input_dim, output_dim, dropout_prob=0.0, split=False):
    if split:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 2*output_dim)
        )
    else:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, output_dim)
        )

def decoder(input_dim, dropout_prob=0.0, split=False):
    if split:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 2*784),
            nn.Sigmoid()
        )
        
    else:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

def beefy_decoder(input_dim, dropout_prob=0.0, split=False):
    if split:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 2*784),
            nn.Sigmoid()
        )
        
    else:
        return nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

class Discriminator_Really_Small(nn.Module):
    def __init__(self, input_dim, multiplier=None, dropout_prob=0.0):
        super(Discriminator_Really_Small, self).__init__()
        if multiplier is None:
            self.multiplier = input_dim
        else:
            self.multiplier = multiplier
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, self.multiplier * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.multiplier * 16, self.multiplier * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.multiplier * 16, self.multiplier * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.multiplier * 16, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        validity = self.model(z)
        return validity

class Discriminator_Small(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_Small, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        validity = self.model(z)
        return validity

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 3*1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(3*1024, 3*1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(3*1024, 3*1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(3*1024, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        validity = self.model(z)
        return validity

class ACCA_Private(nn.Module):
    def __init__(self, z_dim, num_z_inputs, hx_dim, hy_dim, dropout_prob=0.0, encoder_function=encoder, decoder_function=decoder):
        super(ACCA_Private, self).__init__()
        self.z_dim = z_dim
        self.num_z_inputs = num_z_inputs
        self.hx_dim = hx_dim
        self.hy_dim = hy_dim
        self.dropout_prob = dropout_prob
        self.encode_hx = encoder_function(784, hx_dim, dropout_prob)
        self.encode_hy = encoder_function(784, hy_dim, dropout_prob)
        if num_z_inputs == 1:
            self.encode_z = encoder_function(784, z_dim, dropout_prob)
        elif num_z_inputs == 2:
            self.encode_z = encoder_function(784*2, z_dim, dropout_prob)
        self.decode_x = decoder_function(hx_dim + z_dim, dropout_prob)
        self.decode_y = decoder_function(hy_dim + z_dim, dropout_prob)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y): 
        return self.decode(*self.encode(x,y))

    def encode(self, x, y):
        x_input = x.view(x.shape[0],-1)
        y_input = y.view(y.shape[0],-1)
        hx = self.encode_hx(x_input)
        hy = self.encode_hy(y_input)
        if self.num_z_inputs == 1:
            z = self.encode_z(x_input)
        elif self.num_z_inputs == 2:
            z = self.encode_z(torch.cat((x_input,y_input),1))
        return z, hx, hy

    def decode(self, z, hx, hy):
        x_input = torch.cat((z,hx),1)
        y_input = torch.cat((z,hy),1)
        x = self.decode_x(x_input)
        y = self.decode_y(y_input)
        return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)

class VCCA_Private(nn.Module):
    def __init__(self, z_dim, num_z_inputs, hx_dim, hy_dim, dropout_prob=0.0, encoder_function=encoder, decoder_function=decoder):
        super(VCCA_Private, self).__init__()
        self.z_dim = z_dim
        self.hx_dim = hx_dim
        self.hy_dim = hy_dim
        self.num_z_inputs = num_z_inputs
        self.dropout_prob = dropout_prob

        self.encode_hx = encoder_function(784, hx_dim, dropout_prob, split=True)
        self.encode_hy = encoder_function(784, hy_dim, dropout_prob, split=True)
        if num_z_inputs == 1:
            self.encode_z = encoder_function(784, z_dim, dropout_prob, split=True)
        elif num_z_inputs == 2:
            self.encode_z = encoder_function(784*2, z_dim, dropout_prob, split=True)
        self.decode_x = decoder_function(hx_dim + z_dim, dropout_prob, split=False)
        self.decode_y = decoder_function(hy_dim + z_dim, dropout_prob, split=False)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y):
        z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar = self.encode(x, y)
        z = self.reparameterize(z_mu, z_logvar)
        hx = self.reparameterize(hx_mu, hx_logvar)
        hy = self.reparameterize(hy_mu, hy_logvar)
        x_hat, y_hat = self.decode(z, hx, hy)
        return x_hat, y_hat, z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar, z, hx, hy

    def encode(self, x, y):
        x_input = x.view(x.shape[0],-1)
        y_input = y.view(y.shape[0],-1)
        hx_mu, hx_logvar = torch.chunk(self.encode_hx(x_input),2,dim=1)
        hy_mu, hy_logvar = torch.chunk(self.encode_hy(y_input),2,dim=1)
        z_mu, z_logvar = torch.chunk(self.encode_z(x_input),2,dim=1)
        return z_mu, z_logvar, hx_mu, hx_logvar, hy_mu, hy_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, hx, hy):
        x_input = torch.cat((z,hx),1)
        y_input = torch.cat((z,hy),1)
        x = self.decode_x(x_input)
        y = self.decode_y(y_input)
        return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)

def unconstrained_encoder(input_dim, output_dim, dropout_prob=0.0, num_layers=3, head=None, split=False):
    layers = [nn.Dropout(p=dropout_prob), nn.Linear(input_dim, 1024), nn.ReLU()]
    for i in range(num_layers):
        layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(1024,1024))
        layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=dropout_prob))
    if split:
        layers.append(nn.Linear(1024,2*output_dim))
    else:
        layers.append(nn.Linear(1024,output_dim))    
    if head=='softmax':
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

def unconstrained_decoder(input_dim, dropout_prob=0.0, num_layers=3, split=False):
    layers = [nn.Dropout(p=dropout_prob), nn.Linear(input_dim, 1024), nn.ReLU()]
    for i in range(num_layers):
        layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(1024,1024))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(1024, 784))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

class VCCA_Single(nn.Module):
    def __init__(self, z_dim, num_z, dropout_prob=0.0, encoder_function=encoder, decoder_function=decoder):
        super(VCCA_Single, self).__init__()
        self.z_dim = z_dim
        self.dropout_prob = dropout_prob
        self.num_z = num_z

        if num_z == 1:
            self.encode_z = encoder_function(784, z_dim, dropout_prob, split=True)
        elif num_z == 2:
            self.encode_z = encoder_function(784*2, z_dim, dropout_prob, split=True)
        self.decode_x = decoder_function(z_dim, dropout_prob, split=False)
        self.decode_y = decoder_function(z_dim, dropout_prob, split=False)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y):
        z_mu, z_logvar = self.encode(x, y)
        z = self.reparameterize(z_mu, z_logvar)
        x_hat, y_hat = self.decode(z)
        return x_hat, y_hat, z_mu, z_logvar, z

    def encode(self, x, y):
        x_input = x.view(x.shape[0],-1)
        y_input = y.view(y.shape[0],-1)
        if self.num_z == 1:
            z_mu, z_logvar = torch.chunk(self.encode_z(x_input),2,dim=1)
        elif self.num_z == 2:
            z_mu, z_logvar = torch.chunk(self.encode_z(torch.cat((x_input,y_input),1)),2,dim=1)
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.decode_x(z)
        y = self.decode_y(z)
        return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)

class ACCA_Single(nn.Module):
    def __init__(self, z_dim, num_z=2, dropout_prob=0.0, encoder_function=encoder, decoder_function=decoder):
        super(ACCA_Single, self).__init__()
        self.z_dim = z_dim
        self.num_z = num_z
        self.dropout_prob = dropout_prob

        if num_z == 1:
            self.encode_z = encoder_function(784, z_dim, dropout_prob, split=False)
        elif num_z == 2:
            self.encode_z = encoder_function(784*2, z_dim, dropout_prob, split=False)
        self.decode_x = decoder_function(z_dim, dropout_prob, split=False)
        self.decode_y = decoder_function(z_dim, dropout_prob, split=False)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y): 
        return self.decode(*self.encode(x,y))

    def encode(self, x, y):
        x_input = x.view(x.shape[0],-1)
        y_input = y.view(y.shape[0],-1)
        if self.num_z == 1:
            z = self.encode_z(x_input)
        elif self.num_z == 2:
            z = self.encode_z(torch.cat((x_input,y_input),1))
        return z

    def decode(self, z):
        x = self.decode_x(z)
        y = self.decode_y(z)
        return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)

# class ACCA_Complete(nn.Module):
#     def __init__(self, z_dim, num_z_inputs, hx_dim, hy_dim, num_layers=3, dropout_prob=0.0):
#         super(ACCA_Complete, self).__init__()
#         self.z_dim = z_dim
#         self.num_z_inputs = num_z_inputs
#         self.hx_dim = hx_dim
#         self.hy_dim = hy_dim
#         self.num_layers = num_layers
#         self.dropout_prob = dropout_prob
#         self.encode_hx = unconstrained_encoder(784, hx_dim, dropout_prob, num_layers)
#         self.encode_hy = unconstrained_encoder(784, hy_dim, dropout_prob, num_layers)
#         if num_z_inputs == 1:
#             self.encode_z = unconstrained_encoder(784, z_dim, dropout_prob, num_layers, head='softmax')
#         elif num_z_inputs == 2:
#             self.encode_z = unconstrained_encoder(784*2, z_dim, dropout_prob, num_layers, head='softmax')
#         self.decode_x = unconstrained_decoder(hx_dim + z_dim, dropout_prob, num_layers)
#         self.decode_y = unconstrained_decoder(hy_dim + z_dim, dropout_prob, num_layers)
#         self.weight_init()

#     def weight_init(self):
#         for block in self._modules:
#             for m in self._modules[block]:
#                 kaiming_init(m)

#     def forward(self, x, y): 
#         return self.decode(*self.encode(x,y))

#     def encode(self, x, y):
#         x_input = x.view(x.shape[0],-1)
#         y_input = y.view(y.shape[0],-1)
#         hx = self.encode_hx(x_input)
#         hy = self.encode_hy(y_input)
#         if self.num_z_inputs == 1:
#             z = self.encode_z(x_input)
#         elif self.num_z_inputs == 2:
#             z = self.encode_z(torch.cat((x_input,y_input),1))
#         return z, hx, hy

#     def decode(self, z, hx, hy):
#         x_input = torch.cat((z,hx),1)
#         y_input = torch.cat((z,hy),1)
#         x = self.decode_x(x_input)
#         y = self.decode_y(y_input)
#         return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)

# class ACCA_Experiment4(nn.Module):
#     def __init__(self, z_dim, num_z_inputs, hx_dim, hy_dim, num_layers=4, dropout_prob=0.0):
#         super(ACCA_Experiment4, self).__init__()
#         self.z_dim = z_dim
#         self.num_z_inputs = num_z_inputs
#         self.hx_dim = hx_dim
#         self.hy_dim = hy_dim
#         self.num_layers = num_layers
#         self.dropout_prob = dropout_prob
#         self.encode_hx = unconstrained_encoder(784, hx_dim, dropout_prob, num_layers)
#         self.encode_hy = unconstrained_encoder(784, hy_dim, dropout_prob, num_layers)
#         if num_z_inputs == 1:
#             self.encode_z = unconstrained_encoder(784, z_dim, dropout_prob, num_layers, head=None)
#         elif num_z_inputs == 2:
#             self.encode_z = unconstrained_encoder(784*2, z_dim, dropout_prob, num_layers, head=None)
#         self.decode_x = unconstrained_decoder(hx_dim + z_dim, dropout_prob, num_layers)
#         self.decode_y = unconstrained_decoder(hy_dim + z_dim, dropout_prob, num_layers)
#         self.weight_init()

#     def weight_init(self):
#         for block in self._modules:
#             for m in self._modules[block]:
#                 kaiming_init(m)

#     def forward(self, x, y): 
#         return self.decode(*self.encode(x,y))

#     def encode(self, x, y):
#         x_input = x.view(x.shape[0],-1)
#         y_input = y.view(y.shape[0],-1)
#         hx = self.encode_hx(x_input)
#         hy = self.encode_hy(y_input)
#         if self.num_z_inputs == 1:
#             z = self.encode_z(x_input)
#         elif self.num_z_inputs == 2:
#             z = self.encode_z(torch.cat((x_input,y_input),1))
#         return z, hx, hy

#     def decode(self, z, hx, hy):
#         x_input = torch.cat((z,hx),1)
#         y_input = torch.cat((z,hy),1)
#         x = self.decode_x(x_input)
#         y = self.decode_y(y_input)
#         return x.view(x.shape[0],1,28,28), y.view(y.shape[0],1,28,28)
