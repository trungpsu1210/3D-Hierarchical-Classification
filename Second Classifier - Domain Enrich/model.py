import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from utils import *
from Preporcessing_fromH5 import DatasetFromHdf5
import argparse

device = get_default_device()

### Multi Slices model ###
#Bidirectional Convolutional LSTM

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

        # pdb.set_trace()
        reversed_idx = list(reversed(range(xreverse.shape[1])))
        xreverse = xreverse[:, reversed_idx, :, :, :]  # reverse temporal outputs.
        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)
        output_list_cat = torch.cat((y_out_fwd[0], y_out_rev[0]), dim=2)
        return output_list_cat

# DenseNet Decoder 

class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2 * 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3 * 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4 * 32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5 * 32)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_planes + 5 * 32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    # Plus because concatenation
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class DenseNet_Decoder(nn.Module):
    def __init__(self):
        super(DenseNet_Decoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        self.residual_block53 = ResidualBlock(128)
        self.residual_block54 = ResidualBlock(128)
        self.residual_block55 = ResidualBlock(128)
        self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        self.residual_block63 = ResidualBlock(64)
        self.residual_block64 = ResidualBlock(64)
        self.residual_block65 = ResidualBlock(64)
        self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        self.residual_block73 = ResidualBlock(32)
        self.residual_block74 = ResidualBlock(32)
        self.residual_block75 = ResidualBlock(32)
        self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        self.residual_block83 = ResidualBlock(16)
        self.residual_block84 = ResidualBlock(16)
        self.residual_block85 = ResidualBlock(16)
        self.residual_block86 = ResidualBlock(16)
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0) 
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4):

        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        x52 = torch.cat([x5, x1], 1)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.residual_block82(x8)
        x8 = torch.cat([x8, x], 1)
        x9 = self.relu(self.conv_refin(x8))
        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
        reconstruct = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        reconstruct = self.refine3(reconstruct)
        return reconstruct

# DenseNet Encoder 

class DenseNet_Encoder(nn.Module):
  def __init__(self):
      super(DenseNet_Encoder, self).__init__()

      main_model = models.densenet121(pretrained=False)
      self.conv0 = main_model.features.conv0
      self.norm0 = main_model.features.norm0
      self.relu0 = main_model.features.relu0
      self.pool0 = main_model.features.pool0

      self.dense_block1 = main_model.features.denseblock1
      self.trans_block1 = main_model.features.transition1

      self.dense_block2 = main_model.features.denseblock2
      self.trans_block2 = main_model.features.transition2

      self.dense_block3 = main_model.features.denseblock3
      self.trans_block3 = main_model.features.transition3

      self.dense_block41 = main_model.features.denseblock4
      self.norm5 = main_model.features.norm5

      self.dense_block42 = BottleneckDecoderBlock(512, 256)  # 512
      self.trans_block4 = TransitionBlock(768, 128)  # 768
      self.residual_block41 = ResidualBlock(128)
      self.residual_block42 = ResidualBlock(128)
      self.residual_block43 = ResidualBlock(128)
      self.residual_block44 = ResidualBlock(128)
      self.residual_block45 = ResidualBlock(128)
      self.residual_block46 = ResidualBlock(128)

      # Define the decoder for skip connection
      self.Decoder = DenseNet_Decoder()

  def forward(self, x):

    ### Encoder ###
    x0 = self.conv0(x)
    x0 = self.norm0(x0)
    x0 = self.relu0(x0)
    x0 = self.pool0(x0)

    x1 = self.dense_block1(x0)
    x1 = self.trans_block1(x1)

    x2 = self.dense_block2(x1)
    x2 = self.trans_block2(x2)

    x3 = self.dense_block3(x2)
    x3 = self.trans_block3(x3)

    x42 = self.dense_block42(x3)
    x4 = self.trans_block4(x42)

    x41 = self.dense_block41(x3)
    feat_extract = self.norm5(x41)

    ### Decoder ###
    # Perform skip connection
    re_images = self.Decoder(x, x1, x2, x4)

    return feat_extract, re_images

#Final Model 

class Before_cat(nn.Module):

    def __init__(self):
        super(Before_cat, self).__init__()
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d((1)),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.9),
            nn.Linear(256, 32),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

class After_cat(nn.Module):

    def __init__(self):
        super(After_cat, self).__init__()
        self.network = nn.Sequential(
            # 20 slices
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

class Densenet_En_De_BCLSTM(nn.Module):

    def __init__(self):
        super(Densenet_En_De_BCLSTM, self).__init__()
        self.feature_extraction = DenseNet_Encoder()
        self.ConvBLSTM = ConvBLSTM(in_channels=1024, hidden_channels=[512], kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.Classifier_before = Before_cat()
        self.Classifier_after = After_cat()

    def forward(self, input):
        input_tensor_LSTM = torch.zeros((input.shape[0], input.shape[1], 1024, 4, 4))
        reconstruct_images = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]))
        input_tensor_classifier = torch.tensor([])
        input_tensor_LSTM = to_device(input_tensor_LSTM, device)
        input_tensor_classifier = to_device(input_tensor_classifier, device)
        reconstruct_images = to_device(reconstruct_images, device)
        Slice = input.shape[1]
        for step in range(Slice):
            img = input[:, step, :, :, :]
            x, y = self.feature_extraction(img)
            # input_tensor = torch.cat((input_tensor, x), dim = 0)
            input_tensor_LSTM[:, step, :, :, :] = x
            reconstruct_images[:, step, :, :, :] = y

        ## Use ConvBLSTM
        output = self.ConvBLSTM(input_tensor_LSTM, input_tensor_LSTM)
        for step in range(Slice):
            img = output[:, step, :, :, :]
            x = self.Classifier_before(img)
            input_tensor_classifier = torch.cat((input_tensor_classifier, x), dim=1)

        final_out = self.Classifier_after(input_tensor_classifier)

        return final_out, reconstruct_images

### Cross track MIP model ###

class Cross_track_model(nn.Module):

    def __init__(self):
        super(Cross_track_model, self).__init__()

        self.model = models.densenet121(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
          )

    def forward(self, x):
        x1 = self.model(x)
        out = self.classifier(x1)

        return out

### Along track MIP model ###


class Along_track_model(nn.Module):

    def __init__(self):
        super(Along_track_model, self).__init__()

        self.model = models.densenet121(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU()
          )

    def forward(self, x):
        x1 = self.model(x)
        out = self.classifier(x1)

        return out

### Fusion together ###

class Fusion_model(nn.Module):

    def __init__(self):
        super(Fusion_model, self).__init__()

        self.AT_model = Along_track_model()
        self.CT_model = Cross_track_model()
        self.En_DE_LSTM_model = Densenet_En_De_BCLSTM()
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax()
        )

    def forward(self, AT_img, CT_img, MS_img):
        AT_feature = self.AT_model(AT_img)
        CT_feature = self.CT_model(CT_img)
        MS_feature, Re_img = self.En_DE_LSTM_model(MS_img)

        total_feature = torch.cat((AT_feature, CT_feature, MS_feature), dim=1)
        out = self.classifier(total_feature)

        return out, Re_img
