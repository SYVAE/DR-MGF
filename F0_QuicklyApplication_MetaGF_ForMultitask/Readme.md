# Meta-GF-V2

This is a new version of Meta-GF, which is re-written by the authors and can be applied to multi-task learning.

**To run this demo:  ./E1_Multi_task/models/model_segnet_mt_ECCV.py** (The nyuv2 datasets can be downloaded from  https://pan.baidu.com/s/1lEI4ir0l0-MGx-fJt3c88g?pwd=cm3v extraction code: cm3v, and put it at "./data/nyuv2")



### How to replace the vanilla gradient optimizer by Meta-GF?

Please refer to the implementation in  ./E1_Multi_task/models/model_segnet_mt_ECCV.py

1. Firstly, you should organize the multiple outputs of your model into list format, for example:

   ```python
   t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
   t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
   t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
   t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
   
   return [t1_pred, t2_pred, t3_pred], self.logsigma #TAKSNUM=3
   ```

2. Then, define the routing-weights model:

   ```python
   weight_model = Meta_fusion_weights_list(your_model,tasknum=TAKSNUM)#in our settings:your_model=SegNet_MTAN, TAKSNUM=3
   weight_model = torch.nn.DataParallel(weight_model).cuda()
   '''3. Defining the optimizer'''
   
   optimization_paramslist = []
   for i in range(0, TASKNUM):
       optimization_paramslist.append(
           {"params": weight_model.module.weightlist[i].parameters(), "initial_lr": opt.metalr, "lr": opt.metalr})
   ```

3. Defining the Optimizer:

   ```python
   FusionOptimizer = torch.optim.Adam(optimization_paramslist, lr= opt.lr)  # the weight decay matters
   taskmodel_optimizer=optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
   scheduler1 = optim.lr_scheduler.StepLR(taskmodel_optimizer, step_size=100, gamma=0.5)
   optimizer = MetaGrad(optimizer=taskmodel_optimizer, temperature=1,
                        tasknum=TASKNUM,
                        meta_Optimizer=FusionOptimizer, inneriteration=1,device=device)
   lambda_weight = np.ones([3, 200])
   ```

4. We disentangle the training of different tasks, and finally fusing the separate learned task models by meta-weighted gradient fusion. Specifically, in one learning epoch:

   ```python
   weightlist = []
   for i in range(0, TASKNUM):
       weightlist.append([])
   
   oldmodel = deepcopy(SegNet_MTAN)
   for taskid in tqdm(range(0,TASKNUM)):
       train_minibatch(nyuv2_train_loader, SegNet_MTAN, device,
                   optimizer,  weight_model, epoch, taskid, lambda_weight,taskmodel_optimizer)
   
       adaptmodel = deepcopy(SegNet_MTAN)
       weightlist[taskid] = (deepcopy(adaptmodel.state_dict()))
       del adaptmodel
       deepcopy_parameter(SegNet_MTAN, oldmodel)
       # SegNet_MTAN.load_state_dict(oldmodel.state_dict())
   oldweightmodel = deepcopy(weight_model)
   print(">>>>>meta updating")
   adaptionmodel = deepcopy(SegNet_MTAN)
   tmp = optimizer.pc_backward(weightlist, adaptionmodel, weight_model, nyuv2_train_loader, epoch)
   innner_loop_state = deepcopy(tmp)
   print('------------------the total time cost:{}'.format(time.time() - starttime))
   del tmp
   with torch.no_grad():
       for n, p in weight_model.named_parameters():
           p.data = deepcopy(
               (1 - EMAoldmomentum) * p.data + EMAoldmomentum * oldweightmodel.state_dict()[n])
   SegNet_MTAN.load_state_dict(innner_loop_state)
   del adaptionmodel
   print(">>>>>meta updating")
   scheduler1.step()
   # stepSheduler.step()
   ```

5. The independent learning process:**train_minibatch()** is :  

   ```python
   def train_minibatch(train_loader, multi_task_model, device,
                              optimizer,weightmodel,epoch,task,lambda_weight,taskOptimizer):
   
       multi_task_model.train()
       train_dataset = iter(train_loader)
       
       for k in range(train_batch):
           cost = np.zeros(24, dtype=np.float32)
           train_data, train_label, train_depth, train_normal = train_dataset.next()
           train_data, train_label = train_data.to(device), train_label.long().to(device)
           train_depth, train_normal = train_depth.to(device), train_normal.to(device)
   
           train_pred, logsigma = multi_task_model(train_data)
   
           train_loss = Calculating_loss(train_pred,train_label, train_depth, train_normal)# Calculating multi-task loss
   
           loss=0
           for i in range(TASKNUM):
               if i==task:
                   loss=loss+train_loss[i] * lambda_weight[i, epoch]
               else:
                   loss = loss + train_loss[i] * lambda_weight[i, epoch]*AUXILIARY  #Auciliary learning for narrowing the fusion gap
          
           taskOptimizer.zero_grad()
           loss.backward()
           taskOptimizer.step()
           
   ```

6.  In the optimizer.pc_backward process，you should adapt the "adapt_model" function in  "./E1_Multi_task/models/model_segnet_mt_ECCV.py" to your training objectives, i.e, the loss function calculation part:

   ```python
   def adapt_model(self, model, weightmodel, Grad_Dictlist, train_loader, epoch,innner_loop_state):
       # switch to train mode
       self.meta_weightOPtmizer.zero_grad()
       train_batch = len(train_loader)
       train_dataset = iter(train_loader)
       final_state=None
       for k in range(train_batch):
           tmpmodel = deepcopy(model)
           train_data, train_label, train_depth, train_normal = train_dataset.next()
           train_data, train_label = train_data.to(self.device), train_label.long().to(self.device)
           train_depth, train_normal = train_depth.to(self.device), train_normal.to(self.device)
   
           self.meta_fusingGradigent(tmpmodel, weightmodel, Grad_Dictlist, ifdifupdating=True)
            
           ################replace this part with your own training objectives
           train_pred, logsigma = tmpmodel(train_data)
           train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                         model_fit(train_pred[1], train_depth, 'depth'),
                         model_fit(train_pred[2], train_normal, 'normal')]
   
           tmploss = 0
           for i in range(3):
               tmploss = tmploss + train_loss[i]
           #############################################################
               
           self.meta_weightOPtmizer.zero_grad()
           tmploss.backward()
   
           self.meta_weightOPtmizer.step()
           self.meta_weightOPtmizer.zero_grad()
           # del tmpmodel
           '''UPDATING THE BN norm'''
           from collections import OrderedDict
           average_dict = OrderedDict()
           for name in innner_loop_state:
               featurename = name.split('.')[-1]
               if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                   average_dict[name] = deepcopy(tmpmodel.state_dict()[name])
               else:
                   average_dict[name] = deepcopy(innner_loop_state[name])
   
           model.load_state_dict(average_dict)
           innner_loop_state = deepcopy(average_dict)
           final_state = deepcopy(tmpmodel.state_dict())
           del tmpmodel
   
   
       return final_state
   ```
