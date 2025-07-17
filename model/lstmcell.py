import torch
import torch.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias = True, surrogate_fun1 = None, alpha = 3.0, surrogate_fun2 = None):
        super(LSTMCell, self).__init__()
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias = bias)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias = bias)

        self.surrogate_fun1 = surrogate_fun1
        self.surrogate_fun2 = surrogate_fun2
        self.alpha = alpha
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reset_parameters()

    def forward(self, input, hc = None):
        if hc is None:
            h = torch.zeros(size=[input.shape[0], self.hidden_size], dtype = torch.float, device= input.device)
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]

        if self.surrogate_fun2 is None:
            # erf surrogate fun
            i, f, g, o = torch.split(self.surrogate_fun1(self.linear_ih(input) + self.linear_hh(h)),
                                     self.hidden_size, dim=1)

            # act surrogate fun.
            # i, f, g, o = torch.split(self.surrogate_fun1(self.linear_ih(input) + self.linear_hh(h)),
            #                          self.hidden_size, dim=1)
        else:
            i, f, g, o = torch.split(self.linear_ih(input) + self.linear_hh(h), self.hidden_size, dim=1)
            i = self.surrogate_fun1(i)
            f = self.surrogate_fun1(f)
            g = self.surrogate_fun2(g)
            o = self.surrogate_fun2(o)  # reviesed

        c = self.surrogate_fun2(c * f + i * g)
        h = c * o
        return h, c

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)
