import torch
import optuna
import torch.nn as nn
from utils.utils import progress_bar
import matplotlib.pyplot as plt
from utils.utils import pgd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose


class CURELearner():
    '''Strongly modified version of the implementation of "Robustness via curvature regularization, and vice versa"
       in https://arxiv.org/abs/1811.09716. This version includes higher and mixed order regularization and higher

        Args:
            net (Pytorch nn): network structure
            trainloader (PyTorch Dataloader): The train loader 
            testloader (PyTorch Dataloader): The test loader
            device ('cpu' or 'cuda' if GPU available): type of decide to move tensors
            lambda_ (float): power of regularization
            path (string): path to save the best model
            acc ([1, 2, 4, 6, 8]): level of accuracy for the computation of the Hessian vector product'''

    def __init__(self, net, trainloader, testloader,  lambda_0=4, lambda_1=4, lambda_2=0, transformer=None, trial=None, image_min=0, image_max=1, device='cuda',
                 path='./checkpoint', acc=0):

        if not torch.cuda.is_available() and device == 'cuda':
            raise ValueError("cuda is not available")

        self.net = net.to(device)
        if transformer is not None and type(transformer.transforms[0]) == ToTensor:
            self.transformer = Compose(transformer.transforms[1:])
        else:
            self.transformer = transformer

        self.criterion = nn.CrossEntropyLoss()
        self.trial = trial
        self.device = device
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.trainloader, self.testloader = trainloader, testloader
        self.path = path
        self.test_acc_adv_best = 0
        self.image_min = image_min
        self.image_max = image_max
        self.acc = acc
        self.train_loss, self.train_acc = [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean = [], [], []
        self.train_curv_total, self.test_curv_total = [], []
        self.train_curv_0, self.train_curv_1, self.train_curv_2 = [], [], []
        self.test_curv_0, self.test_curv_1, self.test_curv_2 = [], [], []


    def set_optimizer(self, optim_alg='Adam', args={'lr': 1e-4}, scheduler=None, args_scheduler={}):
        '''Set the optimizer of the network
    
        Args:
            optim_alg (string): Name of the optimizer
            args (dict): Parameter of the optimizer
            scheduler (optim.lr_scheduler): Learning rate scheduler
            args_scheduler (dict): Parameters of the scheduler'''

        self.optimizer = getattr(optim, optim_alg)(
            self.net.parameters(), **args)
        if not scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=7, gamma=0.1)
        else:
            self.scheduler = getattr(optim.lr_scheduler, scheduler)(
                self.optimizer, **args_scheduler)

    def train(self, h=[3], epochs=15, epsilon=8/255):
        '''Train the network

        Args:
            h (list): List with length less than the number of epochs. Different h for different epochs of training,
                    can have a single number or a list of floats for each epoch
            epochs (int): Number of epochs'''

        if len(h) > epochs:
            raise ValueError(
                'Length of h should be less than number of epochs')
        if len(h) == 1:
            h_all = epochs * [h[0]]
        else:
            h_all = epochs * [1.0]
            h_all[:len(h)] = list(h[:])
            h_all[len(h):] = (epochs - len(h)) * [h[-1]]

        for epoch, h_tmp in enumerate(h_all):
            self._train(epoch, h=h_tmp)
            self.test(epoch, h=h_tmp, eps=epsilon)

            # This is used for hyperparameter tuning with optuna
            if self.trial is not None:
                current_acc_adv = self.test_acc_adv[-1]
                self.trial.report(current_acc_adv, epoch)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            self.scheduler.step()

    def _train(self, epoch, h):
        '''Train the model'''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        curv, curvature, norm_grad_sum = 0, 0, 0
        curvature_0, curvature_1, curvature_2 = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)
            outputs = self.net.train()(inputs)

            regularizer, grad_norm, curvatures_split_up = self.regularizer(inputs, targets, h=h)

            curvature += regularizer.item()
            curvature_0 += curvatures_split_up[0].item()
            curvature_1 += curvatures_split_up[1].item()
            curvature_2 += curvatures_split_up[2].item()
            loss = self.criterion(outputs, targets)
            loss = loss + regularizer
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            outcome = predicted.data == targets
            num_correct += outcome.sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | curvature: %.3f ' %
                         (train_loss/(batch_idx+1), 100.*num_correct/total, num_correct, total, curvature/(batch_idx+1)))

        self.train_loss.append(train_loss/(batch_idx+1))
        self.train_acc.append(100.*num_correct/total)
        self.train_curv_total.append(curvature/(batch_idx+1))
        self.train_curv_0.append(curvature_0 / (batch_idx + 1))
        self.train_curv_1.append(curvature_1 / (batch_idx + 1))
        self.train_curv_2.append(curvature_2 / (batch_idx + 1))

    def test(self, epoch, h, num_pgd_steps=20, eps=8/255):
        '''Test the model'''
        test_loss, adv_acc, total, curvature, clean_acc, grad_sum = 0, 0, 0, 0, 0, 0
        curvature_0, curvature_1, curvature_2 = 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.testloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net.eval()(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(targets).sum().item()
            total += targets.size(0)

            inputs_pert = pgd(inputs, self.net, epsilon=eps, targets=targets, step_size=0.04, num_steps=num_pgd_steps,
                              normalizer=self.transformer, device=self.device, clip_min=self.image_min, clip_max=self.image_max)

            outputs = self.net(inputs_pert)
            probs, predicted = outputs.max(1)
            adv_acc += predicted.eq(targets).sum().item()
            cur, norm_grad, curvatures_split_up = self.regularizer(inputs, targets, h=h)
            grad_sum += norm_grad
            curvature += cur.item()
            curvature_0 += curvatures_split_up[0].item()
            curvature_1 += curvatures_split_up[1].item()
            curvature_2 += curvatures_split_up[2].item()
            test_loss += cur.item()

        print(f'epoch = {epoch}, adv_acc = {100.*adv_acc/total}, clean_acc = {100.*clean_acc/total}, loss = {test_loss/(batch_idx+1)}',
              f'curvature = {curvature/(batch_idx+1)}')

        self.test_loss.append(test_loss/(batch_idx+1))
        self.test_acc_adv.append(100.*adv_acc/total)
        self.test_acc_clean.append(100.*clean_acc/total)
        self.test_curv_total.append(curvature/(batch_idx+1))
        self.test_curv_0.append(curvature_0 / (batch_idx + 1))
        self.test_curv_1.append(curvature_1 / (batch_idx + 1))
        self.test_curv_2.append(curvature_2 / (batch_idx + 1))
        if self.test_acc_adv[-1] > self.test_acc_adv_best:
            self.test_acc_adv_best = self.test_acc_adv[-1]
            print(f'Saving the best model to {self.path}')
            self.save_model(self.path)

        return test_loss/(batch_idx+1), 100.*adv_acc/total, 100.*clean_acc/total, curvature/(batch_idx+1)

    def _find_z(self, inputs, targets):
        '''Find the direction in the regularizer'''
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(self.net.eval()(inputs), targets)
        loss_z.backward()
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1. * (z+1e-7) / (z.reshape(z.size(0), -
                                           1).norm(dim=1)[:, None, None, None]+1e-7)
        # zero_gradients (on the inputs)
        inputs.grad.detach_()
        inputs.grad.zero_()
        self.net.zero_grad()

        return z, norm_grad

    def _3_diff(self, in_0, in_1, in_2):
        '''Compute third order derivative without dividing by infinitesimal (absorbed by regularizer)'''
        return in_0-2*in_1+in_2

    def regularizer(self, inputs, targets, h=3.):
        '''Taylor regularizer. Includes curvature regularization of accuracy determined by acc property. 
           Also includes gradient and third order regularizers.'''
        acc = self.acc
        z, norm_grad = self._find_z(inputs, targets)

        inputs.requires_grad_()
        outputs_orig = self.net.eval()(inputs)
        loss_orig = self.criterion(outputs_orig, targets)

        reg_0 = torch.Tensor([0]).to(self.device)
        if self.lambda_0 != 0:
            # first order regularization
            first_order = torch.autograd.grad(loss_orig, inputs, create_graph=True)[0].requires_grad_()
            reg_0 = torch.sum(torch.pow(first_order, 2) * self.lambda_0)
            self.net.zero_grad()

        reg_1 = torch.tensor([0]).to(self.device)
        if acc==1:
            outputs_pos = self.net.eval()(inputs + h*z)
            loss_pos = self.criterion(outputs_pos, targets)
            approx = torch.autograd.grad((loss_pos-loss_orig), inputs, create_graph=True)[0]
            pre = approx.reshape(approx.size(0), -1).norm(dim=1)
            reg_1 = torch.sum(pre * self.lambda_1)

        if acc in [2, 4, 6, 8]:
                if acc == 2:
                    # CURE regularization with higher order accuracy O(h^4) instead of O(h^2)
                    # using central finite difference. These coefficients are fixed constants (see https://en.wikipedia.org/wiki/Finite_difference_coefficient)
                    coeffs = torch.tensor([-1/2, +1/2], requires_grad=False).to(self.device)
                    # evaluation points
                    evals = [self.net.eval()(inputs - h*z), self.net.eval()(inputs + h*z)]
                elif acc == 4:
                    # CURE regularization with higher order accuracy O(h^6) instead of O(h^2)
                    coeffs = torch.tensor([1/12, -2/3, 2/3, -1/12], requires_grad=False).to(self.device)
                    # evaluation points
                    evals = [self.net.eval()(inputs - 2*h*z), self.net.eval()(inputs - h*z), self.net.eval()(inputs + h*z), self.net.eval()(inputs + 2*h*z)]
                elif acc==6:
                    # CURE regularization with higher order accuracy O(h^8) instead of O(h^2)
                    coeffs = torch.tensor([-1/60, 3/20, -3/4, 3/4, -3/20, 1/60], requires_grad=False).to(self.device)
                    # evaluation points
                    evals = [self.net.eval()(inputs - 3*h*z), self.net.eval()(inputs - 2*h*z), self.net.eval()(inputs - h*z),
                             self.net.eval()(inputs + h*z), self.net.eval()(inputs + 2*h*z), self.net.eval()(inputs + 3*h*z)]
                elif acc==8:
                    # CURE regularization with higher order accuracy O(h^10) instead of O(h^2)
                    coeffs = torch.tensor([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280], requires_grad=False).to(self.device)
                    # evaluation points
                    evals = [self.net.eval()(inputs - 4*h*z), self.net.eval()(inputs - 3*h*z), self.net.eval()(inputs - 2*h*z),
                             self.net.eval()(inputs - h*z), self.net.eval()(inputs + h*z), self.net.eval()(inputs + 2*h*z),
                             self.net.eval()(inputs + 3*h*z), self.net.eval()(inputs + 4*h*z)]

                losses = torch.stack([self.criterion(ev, targets) for ev in evals])
                lin_comb = torch.sum(coeffs * losses)
                approx = torch.autograd.grad((lin_comb), inputs, create_graph=True)[0]
                pre = approx.reshape(approx.size(0), -1).norm(dim=1)
                reg_1 = torch.sum(pre * self.lambda_1)
        self.net.zero_grad()

        # third order regularization
        reg_2 = torch.Tensor([0]).to(self.device)
        if self.lambda_2 != 0:
            loss_1 = self.criterion(self.net.eval()(inputs - h*torch.ones_like(inputs)), targets)
            loss_2 = self.criterion(self.net.eval()(inputs), targets)
            loss_3 = self.criterion(self.net.eval()(inputs + h*torch.ones_like(inputs)), targets)

            total_fin_dif = self._3_diff(loss_1, loss_2, loss_3)

            third_order_approx = torch.autograd.grad(total_fin_dif, inputs, create_graph=True)[0]
            reg_2 = torch.sum(torch.pow(third_order_approx, 2) * self.lambda_2)
            self.net.zero_grad()

        return (reg_0 + reg_1 + reg_2) / float(inputs.size(0)), norm_grad, [reg_0 / float(inputs.size(0)), reg_1 / float(inputs.size(0)), reg_2 / float(inputs.size(0))]

    def save_model(self, path):
        '''Save the model

        Args:
            path (string): path to save the model'''

        print('Saving...')

        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def save_state(self, path):
        print('Saving...')

        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_acc': self.train_acc,
            'test_acc_clean': self.test_acc_clean,
            'test_acc_adv': self.test_acc_adv,
            'train_curv_total': self.train_curv_total,
            'test_curv_total': self.test_curv_total,
            'train_curv_0': self.train_curv_0,
            'train_curv_1': self.train_curv_1,
            'train_curv_2': self.train_curv_2,
            'test_curv_0': self.test_curv_0,
            'test_curv_1': self.test_curv_1,
            'test_curv_2': self.test_curv_2,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss
        }
        torch.save(state, path)

    def import_model(self, path):
        '''Import the pre-trained model'''
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])

    def import_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.train_acc = checkpoint['train_acc']
        self.test_acc_clean = checkpoint['test_acc_clean']
        self.test_acc_adv = checkpoint['test_acc_adv']
        self.train_curv_total = checkpoint['train_curv_total']
        self.test_curv_total = checkpoint['test_curv_total']
        self.train_curv_0 = checkpoint['train_curv_0']
        self.train_curv_1 = checkpoint['train_curv_1']
        self.train_curv_2 = checkpoint['train_curv_2']
        self.test_curv_0 = checkpoint['test_curv_0']
        self.test_curv_1 = checkpoint['test_curv_1']
        self.test_curv_2 = checkpoint['test_curv_2']
        self.train_loss = checkpoint['train_loss']
        self.test_loss = checkpoint['test_loss']

    def plot_results(self, title=""):
        """Plot the results"""
        plt.figure(figsize=(18, 12))
        plt.suptitle(title + 'Results', fontsize=18, y=0.96)
        plt.subplot(4, 4, 1)
        plt.plot(self.train_acc, Linewidth=2, c='C0')
        plt.plot(self.test_acc_clean, Linewidth=2, c='C1')
        plt.plot(self.test_acc_adv, Linewidth=2, c='C2')
        plt.legend(['train_clean', 'test_clean', 'test_adv'], fontsize=14)
        plt.title('Accuracy', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('epoch', fontsize=14)
        plt.grid()
        plt.subplot(4, 4, 2)
        plt.plot(self.train_curv_total, Linewidth=4, c='black')
        plt.plot(self.train_curv_0, Linewidth=2, c='C0', label='train_curv_0')
        plt.plot(self.train_curv_1, Linewidth=2, c='C1', label='train_curv_1')
        plt.plot(self.train_curv_2, Linewidth=2, c='C2', label='train_curv_2')
        plt.legend(fontsize=14)

        plt.title('Train Curvatures', fontsize=14)
        plt.ylabel('curv', fontsize=14)
        plt.xlabel('epoch', fontsize=14)
        plt.grid()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.subplot(4, 4, 3)
        plt.plot(self.test_curv_total, Linewidth=4, c='black')
        plt.plot(self.test_curv_0, Linewidth=2, c='C0', label='test_curv_0')
        plt.plot(self.test_curv_1, Linewidth=2, c='C1', label='test_curv_1')
        plt.plot(self.test_curv_2, Linewidth=2, c='C2', label='test_curv_2')
        plt.legend(fontsize=14)
        plt.title('Test Curvatures', fontsize=14)
        plt.ylabel('curv', fontsize=14)
        plt.xlabel('epoch', fontsize=14)
        plt.grid()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.subplot(4, 4, 4)
        plt.plot(self.train_loss, Linewidth=2, c='C0')
        plt.plot(self.test_loss, Linewidth=2, c='C1')
        plt.legend(['train', 'test'], fontsize=14)
        plt.title('Loss', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.xlabel('epoch', fontsize=14)
        plt.grid()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
