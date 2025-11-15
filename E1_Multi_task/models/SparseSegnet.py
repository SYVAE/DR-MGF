from tools.Sparse_conv_v2 import *
import torch.nn as nn
from copy import deepcopy

class Conv_layers(nn.Module):
    def __init__(self, channel, pred=False):
        super(Conv_layers, self).__init__()
        self.forwardlist = nn.ModuleList()
        if not pred:
            self.forwardlist.append(SparseConv(in_channels=channel[0], channels=channel[1], kernel_size=3, padding=1))
            self.forwardlist.append(Sparse_BN(num_features=channel[1]))
            self.forwardlist.append(nn.ReLU(inplace=True))
        # else:
        else:
            self.forwardlist.append(SparseConv(in_channels=channel[0], channels=channel[0], kernel_size=3, padding=1))
            self.forwardlist.append(SparseConv(in_channels=channel[0], channels=channel[1], kernel_size=1, padding=0))

    def forward(self, x, fusionweight=None, paramslist=None):
        for layer in self.forwardlist:
            if isinstance(layer, SparseConv):
                if fusionweight is not None and len(paramslist)>0:
                    name = paramslist[0]
                    p = fusionweight[name]
                    x = layer(x, p)  # conv layers
                    paramslist.remove(name)
                    paramslist.remove(name.replace('.weight','.bias')) ##bias is shared
                else:
                    x = layer(x)

            elif isinstance(layer, Sparse_BN):
                if fusionweight is not None and len(paramslist)>0:
                    weightname = paramslist[0]
                    biasname = paramslist[1]
                    weight = fusionweight[weightname]
                    bias = fusionweight[biasname]
                    x = layer(x, weight, bias)

                    paramslist.remove(weightname)
                    paramslist.remove(biasname)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x

class Two_cascaded_convlayers(nn.Module):
    def __init__(self, channel, pred=False, cascaded_num=2):
        super(Two_cascaded_convlayers, self).__init__()
        self.forwardlist = nn.ModuleList()
        for i in range(0, cascaded_num):
            self.forwardlist.append(Conv_layers([channel, channel]))

    def forward(self, x, fusionweight=None, paramslist=None):
        for layer in self.forwardlist:
            x = layer(x, fusionweight, paramslist)

        return x


class SegNet(nn.Module):
    '''encoder--------------------------------------------------------------decoder
         |                                                                     |
         |->task-specific_encoder-attn--->(shared)encoder-attn----->task-specific_decoder-attn-----------taskhead
         |->task-specific_encoder-attn--->(shared)encoder-attn----->task-specific_decoder-attn-----------taskhead
         '''
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # define encoder decoder layers
        self.fixedsharedlayer=[]
        self.encoder_block = nn.ModuleList([Conv_layers([3, filter[0]])])
        self.decoder_block = nn.ModuleList([Conv_layers([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(Conv_layers([filter[i], filter[i + 1]]))
            self.decoder_block.append(Conv_layers([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([Conv_layers([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([Conv_layers([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(Conv_layers([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(Conv_layers([filter[i], filter[i]]))
            else:
                # self.conv_block_enc.append(nn.Sequential(Conv_layers([filter[i + 1], filter[i + 1]]),
                #                                          Conv_layers([filter[i + 1], filter[i + 1]])))
                self.conv_block_enc.append(Two_cascaded_convlayers(filter[i + 1]))

                self.conv_block_dec.append(Two_cascaded_convlayers(filter[i]))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])

        '''shared'''
        self.encoder_block_att = nn.ModuleList([Conv_layers([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([Conv_layers([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(Conv_layers([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(Conv_layers([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(Conv_layers([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(Conv_layers([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = Conv_layers([filter[0], self.class_nb], pred=True)
        self.pred_task2 = Conv_layers([filter[0], 1], pred=True)
        self.pred_task3 = Conv_layers([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, SparseConv):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, Sparse_BN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.get_shared_parameters()

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                # self.encoder_att, self.decoder_att,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def get_shared_parameters(self):
        self.encoder_blocknamelist = []
        self.decoder_blocknamelist = []
        self.conv_block_encnamelist = []
        self.conv_block_decnamelist = []
        self.encoder_block_attnamelist=[]
        self.decoder_block_attnamelist=[]


        for name, p in self.encoder_block.named_parameters():
            self.encoder_blocknamelist.append('encoder_block.' + name)

        for name, p in self.encoder_block_att.named_parameters():
            self.encoder_block_attnamelist.append('encoder_block_att.' + name)

        for index in range(0, 5):
            for name, p in self.decoder_block[-index - 1].named_parameters():
                self.decoder_blocknamelist.append('decoder_block.' + str(5 - index - 1) + '.' + name)

        for index in range(0, 5):
            for name, p in self.decoder_block_att[-index - 1].named_parameters():
                self.decoder_block_attnamelist.append('decoder_block_att.' + str(5 - index - 1) + '.' + name)

        # self.decoder_blocknamelist = self.decoder_blocknamelist[::-1]
        for name, p in self.conv_block_enc.named_parameters():
            self.conv_block_encnamelist.append('conv_block_enc.' + name)

        for index in range(0, 5):
            for name, p in self.conv_block_dec[-index - 1].named_parameters():
                self.conv_block_decnamelist.append('conv_block_dec.' + str(5 - index - 1) + '.' + name)

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                SparseConv(in_channels=channel[0], channels=channel[1], kernel_size=3, padding=1),
                Sparse_BN(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                SparseConv(in_channels=channel[0], channels=channel[0], kernel_size=3, padding=1),
                SparseConv(in_channels=channel[0], channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block


    def att_layer(self, channel):
        att_block = nn.Sequential(
            SparseConv(in_channels=channel[0], channels=channel[1], kernel_size=1, padding=0),
            Sparse_BN(channel[1]),
            nn.ReLU(inplace=True),
            SparseConv(in_channels=channel[1], channels=channel[2], kernel_size=1, padding=0),
            Sparse_BN(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def Obtaining_shallowSharedLayer(self,CONSTANT_SHARELAYER):
        '''the shallow layer: encoder-convblock_enc'''
        for layeidx in range(0,CONSTANT_SHARELAYER):
            self.fixedsharedlayer.append(self.encoder_blocknamelist[0+layeidx*4])
            self.fixedsharedlayer.append(self.conv_block_encnamelist[0+layeidx*4])



    def forward(self, x, fusinoweight=None):
        if fusinoweight is not None:
            paramslist = list(fusinoweight.keys())
        else:
            paramslist = None
        'decoder_block.4.forwardlist.0.bias'
        tmpencoder_blocknamelist = deepcopy(self.encoder_blocknamelist)
        tmpdecoder_blocknamelist = deepcopy(self.decoder_blocknamelist)
        tmpconv_block_encnamelist = deepcopy(self.conv_block_encnamelist)
        tmpconv_block_decnamelist = deepcopy(self.conv_block_decnamelist)
        tmpencoder_block_attnamelist=deepcopy(self.encoder_block_attnamelist)
        tmpdecoder_block_attnamelist=deepcopy(self.decoder_block_attnamelist)

        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i].forward(x, fusinoweight, tmpencoder_blocknamelist)
                g_encoder[i][1] = self.conv_block_enc[i].forward(g_encoder[i][0], fusinoweight,tmpconv_block_encnamelist)
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1], fusinoweight, tmpencoder_blocknamelist)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0], fusinoweight, tmpconv_block_encnamelist)
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i], fusinoweight, tmpdecoder_blocknamelist)
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0], fusinoweight, tmpconv_block_decnamelist)
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i], fusinoweight, tmpdecoder_blocknamelist)
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0], fusinoweight, tmpconv_block_decnamelist)

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:

                    ###self attention
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1],fusinoweight,tmpencoder_block_attnamelist)
                    # atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1],fusinoweight,tmpencoder_block_attnamelist)
                    # atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0],fusinoweight,tmpdecoder_block_attnamelist)
                    # atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',
                                                           align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0],fusinoweight,tmpdecoder_block_attnamelist)
                    # atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma