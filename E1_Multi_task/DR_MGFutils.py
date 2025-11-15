import numpy as np
import time
import torch
import torch.nn.functional as F

from copy import deepcopy
from scipy.optimize import minimize, Bounds, minimize_scalar
"""
Define task metrics, loss functions and model trainer here.
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)





''' ===== multi task MGD trainer ==== '''
def trainer(train_loader, test_loader, multi_task_model, device,
                           optimizer, scheduler, opt,
                           total_epoch=200, method='sumloss', alpha=0.5, seed=0,weight_model=None):
    start_time = time.time()


    rng = np.random.default_rng()
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])

    neg_trace = []
    obj_trace = []
    for index in range(total_epoch):
        epoch_start_time = time.time()
        cost = np.zeros(24, dtype=np.float32)

        # apply Dynamic Weight Average
        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            train_loss_tmp = [0,0,0]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                for i in range(3):
                    train_loss_tmp[i] = train_loss[i] * lambda_weight[i, index]
            else:
                for i in range(3):
                    train_loss_tmp[i] = 1/(2*torch.exp(logsigma[i]))*train_loss[i]+logsigma[i]/2

            optimizer.zero_grad()
            if method == "graddrop":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "mgd":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = mgd(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "pcgrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = pcgrad(grads, rng)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "cagrad":
                for i in range(3):
                    if i < 3:
                        train_loss_tmp[i].backward(retain_graph=True)
                    else:
                        train_loss_tmp[i].backward()
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = cagrad(grads, alpha, rescale=1)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label,  'semantic'),
                             model_fit(test_pred[1], test_depth,  'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = conf_mat.get_metrics()

        scheduler.step()
        if method == "mean":
            torch.save(torch.Tensor(neg_trace), "trace.pt")

        if "debug" in method:
            torch.save(torch.Tensor(obj_trace), f"{method}_obj.pt")

        epoch_end_time = time.time()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23], epoch_end_time-epoch_start_time))
        if "cagrad" in method:
            torch.save(multi_task_model.state_dict(), f"models/{method}-{opt.weight}-{alpha}-{seed}.pt")
        else:
            torch.save(multi_task_model.state_dict(), f"models/{method}-{opt.weight}-{seed}.pt")
    end_time = time.time()
    print("Training time: ", end_time-start_time)
