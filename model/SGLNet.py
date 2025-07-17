import torch.nn as nn
from torch_geometric.nn import EGConv

# from data.structureGraph import *
from model.surrogate_fun import ActFun, Erf, ATan, ATan_P, Erf_P
from model.lstmcell import LSTMCell


# surrogate_function.
# act_enc_fun = ActFun.apply   # encode raw signal to spike train.
# act_fun = ActFun.apply

act_enc_fun = ATan.apply
act_fun = ATan.apply
erf_fun = Erf.apply
erf_p = Erf_P.apply


# membrane potential update, for GCN using edge weight
def mem_update_conv_weight(ops, x, edge_idxs, edge_weight, mem, spike, thresh, lens, decay):
    ops_ret = ops(x, edge_idxs, edge_weight).view(mem.shape)  # mini-batch graph conv,then reshape to each batch.
    mem = mem * decay * (1. - spike) + ops_ret
    spike = act_fun(mem, thresh, lens) # act_fun : approximation firing function
    return mem, spike

def mem_update_conv_weight2(ops, x, edge_idxs, edge_weight, mem, spike, thresh, alpha, decay):
    ops_ret = ops(x, edge_idxs, edge_weight).view(mem.shape)  # mini-batch graph conv,then reshape to each batch.
    mem = mem * decay * (1. - spike) + ops_ret
    spike = act_fun(mem, thresh, alpha) # act_fun : approximation firing function
    return mem, spike

# membrane potential update, for GCN
def mem_update_conv(ops, x, edge_idxs, mem, spike, thresh, lens, decay):
    ops_ret = ops(x, edge_idxs).view(mem.shape)  # mini-batch graph conv,then reshape to each batch.
    mem = mem * decay * (1. - spike) + ops_ret
    spike = act_fun(mem, thresh, lens) # act_fun : approximation firing function
    return mem, spike

# membrane potential update, for GCN
def mem_update_conv2(ops, x, edge_idxs, mem, spike, thresh, alpha, decay):
    ops_ret = ops(x, edge_idxs).view(mem.shape)  # mini-batch graph conv,then reshape to each batch.
    mem = mem * decay * (1. - spike) + ops_ret
    spike = act_fun(mem, thresh, alpha) # act_fun : approximation firing function
    return mem, spike

def mem_update(ops, x, mem, spike, thresh, lens, decay):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem, thresh, lens)
    return mem, spike

def mem_update2(ops, x, mem, spike, thresh, alpha, decay):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem, thresh, alpha)
    return mem, spike


class GraLstmEnc(nn.Module):
    def __init__(self, params, device="cuda:0"):
        super(GraLstmEnc, self).__init__()

        # get params.
        self.cfg_enc = np.array(str.split(params['cfg_enc'], ','), dtype=int)  # encoding scheme transform raw signals to spike trains.
        self.cfg_gnn = np.array(str.split(params['cfg_gnn'], ','), dtype=int)  # gnn_layer(in_channels, out_channels)
        self.cfg_s = np.array(str.split(params['cfg_s'], ','), dtype=int)  # node number.
        self.cfg_fc = np.array(str.split(params['cfg_fc'], ','), dtype=int) # fully connect.
        self.num_classes = int(params['num_classes']) # class number.
        self.num_heads = int(params['num_heads']) # EGC num_heads.
        self.num_bases = int(params['num_bases']) # EGC num_bases
        self.gamma = float(params['gamma']) # dropout coefficient.
        # neuronal threshold,  hyper-parameters of approximate function
        self.thresh, self.lens, self.decay = float(params['thresh']), float(params['lens']), float(params['decay'])
        # encoding step, act_enc_fun parameters.
        self.enc_thresh, self.enc_lens = float(params['enc_thresh']), float(params['enc_lens'])

        # define the module.
        in_enc_planes, out_enc_planes = self.cfg_enc # encoding feature dim
        in_planes, out_planes = self.cfg_gnn # graph conv feature dim.
        self.encoding = nn.Linear(in_enc_planes, out_enc_planes)  # linear transformation to encode, behind with non-linear activation.
        self.conv1 = EGConv(in_planes, out_planes, num_heads= self.num_heads, num_bases=self.num_bases)

        self.lstm_cell1 = LSTMCell(input_size = self.cfg_s[-1] * out_planes, hidden_size = int(self.cfg_fc[0]), surrogate_fun1 = erf_p, alpha=2.0, surrogate_fun2=erf_fun)
        self.fc1 = nn.Linear(self.cfg_fc[0], self.cfg_fc[1])
        self.fc2 = nn.Linear(self.cfg_fc[1], self.num_classes)
        self.device = device

    def forward(self, input):
        x_data, edge_data = input.x, input.edge_index   # min-batch method.
        data = x_data.to(self.device)  #[batch_size, channel, fet_dim, time_win]
        sizes = data.size()
        time_window = sizes[-1]

        # mini-batch sample number.
        sample_num = int(sizes[0] / self.cfg_s[0])     # each samle is 64 channels, so all the channels divide 64 equal the number of sample.
        enc_rest = []
        output_spike = []; output_mem = []
        enc_mem = enc_spike = enc_sumspike = torch.zeros(sample_num * self.cfg_s[0], self.cfg_enc[1], device = self.device) # enc.

        c1_mem = c1_spike = torch.zeros(sample_num, self.cfg_s[0], self.cfg_gnn[1], device=self.device) # graph conv.
        lh1_spike = torch.zeros(sample_num, self.cfg_fc[0], dtype = torch.float, device = self.device)  # lstm h,
        lc1_spike = torch.zeros(sample_num, self.cfg_fc[0], dtype = torch.float, device = self.device)  # lstm c,

        h1_mem = h1_spike = h1_sumspike  = torch.zeros(sample_num, self.cfg_fc[1], device=self.device)  # fc1
        h2_mem = h2_spike = h2_sumspike  = torch.zeros(sample_num, self.num_classes, device=self.device)  # fc2

        inputs = data.split(1, dim=len(sizes)-1)
        for step in range(time_window):   # simulation time steps
            # 1. prepare the data.
            # x = inputs[step].squeeze().unsqueeze(dim=1)  # squeeze the last dimension. for 1 dimentional feature case.
            x = inputs[step].squeeze()

            # data, edge put to device.
            x = x.to(self.device)
            edge_idxs = edge_data.to(self.device)

            # encode raw signal to spike trains.
            # x = act_enc_fun(self.encoding(x), self.enc_thresh, self.enc_lens)  # only using act_fun
            enc_mem, enc_spike = mem_update2(self.encoding, x, enc_mem, enc_spike, self.enc_thresh, self.enc_lens, self.decay)  # using lif neuro.
            x = enc_spike
            # enc_rest.append(x.cpu().numpy())
            enc_sumspike += x

            # 2. graph conv.
            c1_mem, c1_spike = mem_update_conv2(self.conv1, x, edge_idxs, c1_mem, c1_spike, self.thresh, self.lens, self.decay)

            # 3. lstm conv.
            x = c1_spike
            x = x.view(sample_num, -1)
            lh1_spike, lc1_spike = self.lstm_cell1(x, hc = (lh1_spike, lc1_spike))

            # 4. fc step.
            h1_mem, h1_spike = mem_update2(self.fc1, lh1_spike, h1_mem, h1_spike, self.thresh, self.lens, self.decay)
            h1_sumspike += h1_spike

            h2_mem, h2_spike = mem_update2(self.fc2, h1_spike, h2_mem, h2_spike, self.thresh, self.lens, self.decay)
            h2_sumspike += h2_spike

            # record output spikes and mems.
            # output_mem.append(h2_mem.cpu().numpy())
            # output_spike.append(h2_spike.cpu().numpy())

        outputs = h2_sumspike / time_window
        # return outputs, enc_sumspike, h1_sumspike, enc_rest, output_mem, output_spike  # out, encoder_init_feats, hidden_feats, encoder_result.
        return outputs  # only output the outputs.

