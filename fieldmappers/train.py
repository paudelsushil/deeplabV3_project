from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
from .utils import *
from .losses import *

def train(train_data, model, criterion, optimizer, scheduler, train_loss=[], 
          gpu=True):
    """
    Train model

    Params:
        train_data : ''DataLoader''
            Batch grouped data
        model: 
            Model to train
        criterion: 
            Function to caculate loss
        oprimizer: 
            Function for optimzation
        scheduler: 
            Update policy for learning rate decay.
        train_loss: : empty list
            To record average loss for each epoch
        gpu : bool,optional
            Decide whether to use GPU, default is True

    """

    def disable_running_stats(model):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def enable_running_stats(model):
        def _enable(module):
            if (isinstance(module, _BatchNorm)
                and hasattr(module, "backup_momentum")):
                module.momentum = module.backup_momentum

        model.apply(_enable)
    
    model.train()

    # mini batch iteration
    epoch_loss = 0
    i = 0
    
    # get optimizer name
    optimizer_name = optimizer.__class__.__name__
    print(f"Using {optimizer_name}")    

    for img, label, mask in train_data:

        # forward
        img = Variable(img)
        label = Variable(label)
        if gpu:
            img = img.cuda()
            label = label.cuda()

        # Conditional to enable SAM optimizer to run
        if optimizer_name == "SAM":
            enable_running_stats(model)  # Enable running stats

            def closure():
                predictions = model(img)
                loss = eval(criterion)(predictions, label)
                loss.mean().backward()
                return loss

            loss = eval(criterion)(model(img), label)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure)

            disable_running_stats(model)  # Disable running stats

            # second forward-backward step
            def closure2():
                predictions = model(img)
                loss = eval(criterion)(predictions, label)
                loss.mean().backward()
                return loss

            loss2 = eval(criterion)(model(img), label)
            loss2.mean().backward()
            optimizer.second_step(zero_grad=True)

            if any(p.grad is not None for p in model.parameters()):
                optimizer.step(closure2)

            epoch_loss += loss.item()
            i += 1

        else:
            out = model(img)
            label = label * mask.cuda()
            mask = torch.stack([mask]*out.size()[1], dim=1)
            out = out * mask.cuda()

            loss = eval(criterion)(out, label)
            epoch_loss += loss.item()
            i += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # avoid calling to config.yaml
        isCyclicLR = False
        if type(scheduler) == torch.optim.lr_scheduler.CyclicLR:
            scheduler.step()
            isCyclicLR = True

    print(f'train loss:{epoch_loss / i}')
    if isCyclicLR:
        print(f"LR: {scheduler.get_last_lr()}")

    if train_loss != None:
        train_loss.append(float(epoch_loss / i))
