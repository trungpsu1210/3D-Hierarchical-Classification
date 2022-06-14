import torch.nn as nn
import torch
import numpy as np
from torchvision import models
from torchsummary import summary
from utils import *

device = get_default_device()

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        ### input gate
        i = torch.sigmoid(cc_i)
        ### forget gate
        f = torch.sigmoid(cc_f)
        ### output gate
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        ### cell output
        c_next = f * c_cur + i * g
        ### hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvBLSTM(nn.Module):

    # Constructor
    def __init__(self, in_channels, hidden_channels,
                 kernel_size, num_layers, bias=True, batch_first=True):
        super(ConvBLSTM, self).__init__()
        self.forward_net = ConvLSTM(in_channels, hidden_channels, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)
        self.reverse_net = ConvLSTM(in_channels, hidden_channels, kernel_size,
                                    num_layers, batch_first=batch_first, bias=bias)
        self.conv = nn.Conv2d(2 * hidden_channels[-1], 1, kernel_size=1)

    def forward(self, xforward, xreverse):

        """
        xforward, xreverse = B T C H W tensors.
        """

        reversed_idx = list(reversed(range(xreverse.shape[1])))
        xreverse = xreverse[:, reversed_idx, :, :, :]  # reverse temporal outputs.
        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)
        output_list_cat = torch.cat((y_out_fwd[0], y_out_rev[0]), dim=2)

        return output_list_cat

class Densenet(nn.Module):

    def __init__(self):
        super(Densenet, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.network = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, images):
        return self.network(images).squeeze(1)

class Classifier_before_cat(nn.Module):

    def __init__(self):
        super(Classifier_before_cat, self).__init__()
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.9),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

class Classifier_after_cat(nn.Module):

    def __init__(self):
        super(Classifier_after_cat, self).__init__()
        self.network = nn.Sequential(   
            # 20 frames
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax()
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

class Densenet_LSTM(nn.Module):

    def __init__(self):
        super(Densenet_LSTM, self).__init__()
        self.feature_extraction = Densenet()
        self.ConvBLSTM = ConvBLSTM(in_channels=1024, hidden_channels=[512], kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.ConvLSTM = ConvLSTM(input_dim=1024, hidden_dim=[1024], kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.Classifier_before = Classifier_before_cat()
        self.Classifier_after = Classifier_after_cat()

    def forward(self, input):
        input_tensor_LSTM = torch.zeros((input.shape[0], input.shape[1], 1024, 3, 3))
        input_tensor_classifier = torch.tensor([])
        input_tensor_LSTM = to_device(input_tensor_LSTM, device)
        input_tensor_classifier = to_device(input_tensor_classifier, device)
        Slice = input.shape[1]
        for step in range(Slice):
            img = input[:, step, :, :, :]
            x = self.feature_extraction(img)
            # input_tensor = torch.cat((input_tensor, x), dim = 0)
            input_tensor_LSTM[:, step, :, :, :] = x

        # Use Bidirectional Convolutional LSTM
        output = self.ConvBLSTM(input_tensor_LSTM, input_tensor_LSTM)

        for step in range(Slice):
            img = output[:, step, :, :, :]
            x = self.Classifier_before(img)
            input_tensor_classifier = torch.cat((input_tensor_classifier, x), dim=1)

        final_out = self.Classifier_after(input_tensor_classifier)

        return final_out
