import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import ConvLayer, ResidualBlock
import numpy as np

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def skip_sum(x1, x2):
    return x1 + x2

class UNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1,
                 skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2,
                 norm=None, use_upsample_conv=True, with_activation=True, sn=False, multi=False):
        super(UNet, self).__init__()

        self.sn = sn
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        self.num_encoders = num_encoders

        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.activation_name = self.activation
        if self.activation is not None:
            self.activation = getattr(torch, self.activation, 'sigmoid') # 1.object 2.attribute 3.default

        # Build layers
        # N x C x H x W -> N x 32 x H x W
        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=3, stride=1, padding=1, sn=False)  
        self.multi = multi

        self.with_activation = with_activation

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer()

        if multi:
            self.pred_layers = self.build_multiscale_prediction_layers()
            
        self.init_weights()

    def __str__(self):
        summary = "UNet Architecture\n"
        summary += "Total Parameters - {}\n".format(utils.num_trainable_parameters(self))
        return summary
    
    def init_weights(self): # 卷积与内卷积层，凯明初始化权重，偏置如果存在，初始化为0；bn层，权重初始化为1，bias初始化为0.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d): 
                nn.init.kaiming_normal_(m.weight, 10.) 
                #nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_encoders(self):
        encoder_input_sizes = []
        for i in range(self.num_encoders):
            encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) \
                                for i in range(self.num_encoders)]
        
        encoders = nn.ModuleList() #没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言
        for input_size, output_size in zip(encoder_input_sizes, encoder_output_sizes):  #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
                                                                                        #然后返回由这些元组组成的列表。
            encoders.append(ConvLayer(input_size, output_size, kernel_size=3,
                                      stride=2, padding=1, norm=self.norm, sn=False))

        return encoders
                
    def build_resblocks(self):    #输入经encoder编码的数据，通道数达到encoder输出通道数，整个过程通道数，图片大小不变，使用谱归一化
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(ResidualBlock(self.max_num_channels,
                                           self.max_num_channels,
                                           norm=self.norm,
                                           sn=self.sn))
        return resblocks

    def build_prediction_layer(self):
        pred = ConvLayer(self.base_num_channels,
                         self.num_output_channels,
                         kernel_size=1,
                         padding=0,
                         norm=None,
                         sn=None,
                         activation=self.activation_name)
        return pred
                
    def build_decoders(self):  #decoder各层输入通道数
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i+1) \
                                             for i in range(self.num_encoders)]))

        decoders = nn.ModuleList()

        first_layer = True
        for input_size in decoder_input_sizes:
            layer_input = input_size if self.skip_type=='sum' else int(1.5*input_size)
            if not first_layer and self.multi:
                layer_input += self.num_output_channels

            decoders.append(ConvLayer(layer_input,
                                      input_size // 2,
                                      kernel_size=3, stride=1, padding=1,
                                      norm=self.norm, sn=self.sn))
            first_layer = False
        return decoders

    def build_multiscale_prediction_layers(self): # decoder每一层均给一个输出
        pred_sizes = list(reversed([self.base_num_channels * pow(2, i) \
                                             for i in range(self.num_encoders)]))
        
        pred_layers = nn.ModuleList()
        for input_size in pred_sizes:
            pred_layers.append(ConvLayer(input_size,
                                         self.num_output_channels,
                                         kernel_size=1,
                                         padding=0,
                                         norm=None,
                                         sn=None,
                                         activation=self.activation_name))
        return pred_layers
    
    def forward(self, x, flow=False):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # head
        x = self.head(x)
        
        skip_connections = []
        # encoder
        for i, encoder in enumerate(self.encoders):
            skip_connections.append(x)  # 记录encoder每层输出，encoder最后一层输出不记
            x = encoder(x)


        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # flow input from the encoder
        flow_input = x * 1.0
        skip_connections = list(reversed(skip_connections))
        # decoder
        all_pred = []
        for i, (skip_connection, decoder) in enumerate(zip(skip_connections, self.decoders)):
            #x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))
            x = f.interpolate(x, size=(skip_connection.shape[2], skip_connection.shape[3]),
                    mode='nearest')
            x = self.apply_skip_connection(x, skip_connection)
            x = decoder(x)
            if self.multi:
                all_pred.append(self.pred_layers[i](x))
                x = self.apply_skip_connection(x, all_pred[-1]) # 将该层输出与decoder结果级联起来，之后再与encoder层结果级联，作为下一层输入

        if self.multi:
            return all_pred

        # prediction of the last layer
        final_pred = self.pred(x)
        '''
        # flow
        if flow:
            for i, flow_layer in enumerate(self.flow_layers):
                flow_input = flow_layer(self.apply_skip_connection(
                    flow_input,
                    blocks[self.num_encoders - i - 1]))

            flow_pred = self.pred_flow(self.apply_skip_connection(x, head))
            return [img], flow_pred
        else:
            return [img]
        '''
        return [final_pred]
