import torch
import torch.nn as nn
import math


def heaviside(x):
    return (x >= 0.).to(x)

class Erf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha = 2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (- ((ctx.saved_tensors[0]) * ctx.alpha).pow(2)).exp() * (ctx.alpha / math.sqrt(math.pi))

        return grad_x, None

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh = 0.5, lens = 0.5):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.lens = lens
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.lens
        return grad_input * temp.float(), None, None

# defi arc tangent function as surrogate function.
class ATan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, thresh=0.5, alpha = 3.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.thresh = thresh
            ctx.alpha = alpha
        return x.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * (ctx.saved_tensors[0] - ctx.thresh)).pow_(2)) * grad_output

        return grad_x, None, None


# defi arc tangent function as surrogate function.
class ATan_P(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha = 3.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None, None


class Erf_P(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha = 2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return torch.erfc_(-alpha * x) / 2

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (- ((ctx.saved_tensors[0]) * ctx.alpha).pow(2)).exp() * (ctx.alpha / math.sqrt(math.pi))

        return grad_x, None