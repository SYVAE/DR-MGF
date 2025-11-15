#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from D0_dataset.dataloader   import *
from config.args import arg_parser
import models
from tools.Train_utils import *
from models.SDN_Constructing import SDN

args = arg_parser.parse_args()
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
from G0_Gradient_tools.bk.MetaGFgradBN_new import *
color=['cyan','green','blue','olive','black','yellow','red']
def main():
    global args
    IFRECORDING_GRADIENT=False
    print("2022.10.12  EMA-MetaGF---bilevel...joint")
    '''joint means that we jointly train the model in the inference stage'''
    print("MetaGF lr:{0}, Auxilr:{1} EMA_old:{2} Temperature:{3}".format(args.Metalr,args.auxiliarylr,args.EMAoldmomentum,args.temperature))
    time.sleep(5)
    args.sparsification =1






    print("seed:{0}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print("model_savepath:{0}".format(args.save))
    best_err1, best_epoch = 0., 0
    if args.data.startswith('cifar'):
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224


    if args.usingsdn:
        model = SDN(args)
    else:
        model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # define optimizer
    # Defining the weight model
    weight_model = Meta_fusion_weights_list(model, args)
    weight_model = torch.nn.DataParallel(weight_model).cuda()
    '''3. Defining the optimizer'''
    weightdecaylist=[args.metaWeightdecay,args.metaWeightdecay,args.metaWeightdecay,args.metaWeightdecay,args.metaWeightdecay,args.metaWeightdecay,args.metaWeightdecay]
    optimization_paramslist = []
    for i in range(0, args.nBlocks):
        optimization_paramslist.append(
            {"params": weight_model.module.weightlist[i].parameters(), "initial_lr": args.Metalr, "lr": args.Metalr,
             "weight_decay": weightdecaylist[i]})
        # optimization_paramslist.append(
        #     {"params": weight_model.module.thresholdlist[i].parameters(), "initial_lr": 0, "lr": 0,
        #      "weight_decay": 1e-4})
        # optimization_paramslist.append(
        #     {"params": weight_model.module.lrnlist[i].parameters(), "initial_lr": 1e-3, "lr": 1e-3,
        #      "weight_decay": 0})
    stepsizeOptimizer = torch.optim.SGD(optimization_paramslist, lr=args.Metalr,momentum=0.9)  # the weight decay matters
    Meta_OPtimizer = torch.optim.SGD(optimization_paramslist, lr=args.Metalr, momentum=0.9)  # the weight decay matters


    optimizer = MetaGrad(torch.optim.SGD(model.parameters(), args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay), temperature=args.temperature,
                         tasknum=args.nBlocks, criterion=criterion,
                         meta_Optimizer=stepsizeOptimizer, inneriteration=1, args=args,fusionoptimizer=Meta_OPtimizer)
    stepSheduler = torch.optim.lr_scheduler.MultiStepLR(stepsizeOptimizer,
                                                        [int(0.5 * args.epochs), int(0.75 * args.epochs)], gamma=0.1,
                                                        last_epoch=-1)
    # optionally resume from a checkpoint
    cudnn.benchmark = True



    for epoch in range(args.start_epoch, args.epochs):
        for param_group in optimizer.meta_weightOPtmizer.param_groups:
            print("epoch:{0} lr:{1}".format(param_group['lr'],epoch))
            break
        stepSheduler.step()


if __name__ == '__main__':
    # torch.cuda.set_device(2)
    main()