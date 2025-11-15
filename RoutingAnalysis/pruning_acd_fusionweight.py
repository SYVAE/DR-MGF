#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from D0_dataset.dataloader import get_dataloaders
from config.args import arg_parser
from tools.utils import *

args = arg_parser.parse_args()
args.nBlocks =7
args.stepmode = "even"
args.step = 2
args.base = 4
args.grFactor = "1-2-4"
args.bnFactor = "1-2-4"
args.growthRate = 16
args.nChannels = 16
args.data = 'cifar10'
args.arch="msdnet_ge"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tools.Train_utils import *
from G0_Gradient_tools.bk.MetaGFgrad import Meta_fusion_weights_list
from models.SDN_Constructing import SDN
torch.manual_seed(args.seed)

'''Test the performance of the proposed approach when pruning the networks with the learned fusion weight'''

# global args
args.save="/home/sunyi/MetaGF_TPAMI/tmpsave/Metalr0.1_Auxlr0.1_EMAold0.1_t1_bilevel2022-9-14-12-30-47/"
def main(pruningdepth):
    # ['vgg16','resnet56','wideresnet32_4','mobilenet']
    args.evaluate_from = args.save+"/save_models/best_model.pth.tar"
    args.sdnarch = 'resnet56'
    args.data = 'cifar10'
    args.task = args.data
    temperature=args.temperature
    args.data_root= "../data/cifar/"
    print(args.evaluate_from)
    time.sleep(2)
    args.evalmode='anytime'

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    # model = getattr(models, args.arch)(args)
    # n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)

    # model = getattr(models, args.arch)(args)
    model = SDN(args)

    model.cuda()
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    weight_model = Meta_fusion_weights_list(model, args)
    weight_model = torch.nn.DataParallel(weight_model).cuda()


    cudnn.benchmark = True
    '''dataloader...'''
    train_loader, val_loader, test_loader = get_dataloaders(args)
    '''loading task model and weight model'''
    savedict= torch.load(args.evaluate_from,map_location="cuda:0")
    state_dict=savedict['state_dict']
    routing_dict = savedict['routing_index']
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    weight_model.load_state_dict(routing_dict)


    '''0. obtaining the relative importance of parameters to each exit'''
    from collections import OrderedDict
    shared_matrix_dict=OrderedDict()
    with torch.no_grad():
        for name in model.state_dict():
            '''The shared layer:'''
            featurename = name.split('.')[-1]
            if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                continue
            count = 0
            for i in range(0, len(weight_model.module.weightlist)):
                if weight_model.module.weightlist[i].gradlist.__contains__(name):
                    count += 1
            if count == 0:
                continue
            shared_parameter=model.state_dict()[name]

            channel_number=shared_parameter.shape[0]
            '''layernorm'''
            tmp_sumlist = []
            for i in range(0, len(weight_model.module.weightlist)):
                if weight_model.module.weightlist[i].gradlist.__contains__(name):
                    w_ = (getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w).view(-1)
                    w = torch.sigmoid(temperature * w_)
                    tmp_sumlist.append(w.cpu().detach().numpy())

            '''NxM ,N is the number of task and M means the number of the shared channel'''
            shared_matrix=np.stack(tmp_sumlist)
            shared_matrix_dict[name]=shared_matrix

    '''implementing route pruning'''
    file = args.save + "/anytime.txt"
    previous_acc = np.loadtxt(file)
    previous_acc=previous_acc[:,0]
    with torch.no_grad():
        pruningcount=0
        total_number=0
        for name,p in model.named_parameters():
            featurename = name.split('.')[-1]
            if featurename in ["running_mean", "running_var", "num_batches_tracked"] or name.__contains__("classifier"):
                # print(name)
                continue

            if weight_model.module.weightlist[pruningdepth].gradlist.__contains__(name):
                shared_matrix = shared_matrix_dict[name]
                pruning_exit_weight = getattr(weight_model.module.weightlist[pruningdepth], name.replace(".", "#")).w
                N = shared_matrix.shape[0]
                if N < 2:
                    continue
                idx = pruningdepth - (args.nBlocks - N)
                # sortidx=np.argmax(shared_matrix,axis=0)
                sortres = np.argsort(shared_matrix, axis=0)
                maxdepth = sortres[-1]
                second_depth = sortres[-2]

                channelrange = np.arange(0, shared_matrix.shape[1])
                max_value = shared_matrix[maxdepth, channelrange]
                second_value = shared_matrix[second_depth, channelrange]

                mask = maxdepth == idx
                ##resnet
                # probmask = (max_value / second_value) > 100
                probmask = (max_value / second_value) > 1.1
                avg_prob = 1.0 / N
                '''pruning threshold'''

                mask = torch.from_numpy(probmask).cuda() * torch.from_numpy(mask).cuda()
                if p.dim()>=2:
                    outc,inc=p.shape[0],p.shape[1]
                    mask=mask.view(outc,inc)
                    total_number += p.shape[0]*p.shape[1]
                elif p.dim()==1:
                    total_number += p.shape[0] * 1
                    pass
                p[mask] = 0
                pruningcount += mask.sum()

            else:
                pass
    print("pruning ratio:{0}".format(pruningcount / total_number))
    criterion = nn.CrossEntropyLoss().cuda()
    losses, top1,=validate(test_loader, model, criterion)
    reduce_ratio=[]
    for i in range(0,len(top1)):
        print((top1[i].avg-previous_acc[i])/previous_acc[i])
        reduce_ratio.append((top1[i].avg-previous_acc[i])/previous_acc[i])

    return reduce_ratio


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output,_ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     print('Epoch: [{0}/{1}]\t'
            #           'Time {batch_time.avg:.3f}\t'
            #           'Data {data_time.avg:.3f}\t'
            #           'Loss {loss.val:.4f}\t'
            #           'Acc@1 {top1.val:.4f}\t'
            #           'Acc@5 {top5.val:.4f}'.format(
            #         i + 1, len(val_loader),
            #         batch_time=batch_time, data_time=data_time,
            #         loss=losses, top1=top1[-1], top5=top5[-1]))

    # filename="anytime.txt"
    # with open(filename,"w") as f:
    #     for j in range(args.nBlocks):
    #         print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    #         f.write("{0} {1}\n".format(top1[j].avg, top5[j].avg))
    #     print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1


if __name__ == '__main__':
    matrix=np.zeros([7,7])
    for i in range(0,7):
        reduce_ratio=main(pruningdepth=i)
        matrix[i,:]=np.stack(reduce_ratio)[:]
    np.save("matrix",matrix)
    matrix=np.load("matrix.npy")
    plt.figure("relation")
    gain=100
    plt.imshow(np.exp(gain*(-matrix))/np.sum(np.exp(gain*(-matrix)),axis=1,keepdims=True))
    plt.colorbar()
    plt.savefig(args.save+"relation.png")
    plt.show()
    print(matrix)
