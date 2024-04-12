import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

import sys
import os


# Models
from .imagenet1k.models.imagenet1k_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .imagenet1k.models.imagenet1k_vgg import VGG11, VGG13, VGG16, VGG19
from .imagenet1k.models.imagenet1k_densenet import DenseNet_CIFAR, DenseNet121, DenseNet161, DenseNet169, DenseNet201
from .imagenet1k.models.imagenet1k_efficientnet import EfficientNetB0
from .imagenet1k.models.imagenet1k_googlenet import GoogLeNet
from .imagenet1k.models.imagenet1k_resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from .imagenet1k.models.imagenet1k_mobilenet import MobileNet
from .imagenet1k.models.imagenet1k_mobilenetv2 import MobileNetV2
from .imagenet1k.models.imagenet1k_dla import DLA
from .imagenet1k.models.imagenet1k_dpn import DPN26, DPN92
from .imagenet1k.models.imagenet1k_preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .imagenet1k.models.imagenet1k_dla_simple import SimpleDLA
from .imagenet1k.models.imagenet1k_regnet import RegNetX_200MF, RegNetX_400MF, RegNetY_400MF

from .custom_dataset_loaders.desired_imagenet_data import DesiredImageNetData


def _get_model_architecture(model_name, layer_number, out_features):
    if model_name == 'resnet':
        model_class_name = f"ResNet{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"ResNet{layer_number} not found")

    elif model_name == 'vgg':
        model_class_name = f"VGG{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"VGG{layer_number} not found")

    elif model_name == 'densenet':
        model_class_name = f"DenseNet{layer_number}"
        if 'cifar' in str(layer_number):
            return DenseNet_CIFAR(), 'DenseNet_CIFAR'
        elif hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"DenseNet{layer_number} not found")

    elif model_name == 'efficientnet':
        return EfficientNetB0(num_classes=out_features), 'EfficientNetB0'

    elif model_name == 'googlenet':
        model_class_name = f"GoogLeNet"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"GoogleNet not found")

    elif model_name == 'resnext29':
        model_class_name = f"ResNeXt29{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"ResNeXt29{layer_number} not found")

    elif model_name == 'mobilenet':
        return MobileNet(num_classes=out_features), 'MobileNet'

    elif model_name == 'mobilenetv2':
        return MobileNetV2(num_classes=out_features), 'MobileNetV2'

    elif model_name == 'dla':
        return DLA(num_classes=out_features), 'DLA'

    elif model_name == 'dpn':
        model_class_name = f"DPN{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"DPN{layer_number} not found")

    elif model_name == 'preactresnet':
        model_class_name = f"PreActResNet{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"PreActResNet{layer_number} not found")

    elif model_name == 'simpledla':
        return SimpleDLA(num_classes=out_features), 'SimpleDLA'

    elif model_name == 'regnetx':
        model_class_name = f"RegNetX{layer_number}"
        if hasattr(sys.modules[__name__], model_class_name):
            return getattr(sys.modules[__name__], model_class_name)(num_classes=out_features), model_class_name
        else:
            raise ValueError(f"RegNetX{layer_number} not found")

    elif model_name == 'regnety':
        return RegNetY_400MF(num_classes=out_features), 'RegNetY_400MF'

    else:
        raise ValueError("Invalid model name")


def _get_transformations():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def _get_train_test_loaders(batch_size):

    train_transform, test_transform = _get_transformations()
    task_specific_dataset = DesiredImageNetData(num_images_per_class=500, transform=train_transform)

    train_dataset, test_dataset = train_test_split(task_specific_dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class TrainModel(nn.Module):
    def __init__(self, model_name, layer_number='', out_features=10, initialize_weights=False):
        super(TrainModel, self).__init__()
        self.model_name = model_name
        self.layer_number = layer_number
        self.model, self.save_model_name = _get_model_architecture(model_name, layer_number, out_features)
        if initialize_weights:
            self._initialize_weights()
        self.train_transform, self.test_transform = _get_transformations()
        self.batch_size = 128
        self.device = 'cuda'

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.best_accuracy = 0.0
        self.train_acc = 0.0
        self.test_acc = 0.0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _progress_bar(self, current, total, message=''):
        bar_length = 50
        progress = float(current) / total
        arrow = '-' * int(round(progress * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        acc_message = 'Train Acc: {:.2f}% | Test Acc: {:.2f}%'.format(self.train_acc, self.test_acc)

        sys.stdout.write(
            '\r[{0}] {1}/{2} {3}% {4} {5}'.format(arrow + spaces, current, total, int(progress * 100), message,
                                                  acc_message))
        sys.stdout.flush()

    def _test_model(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                self.test_acc = 100. * correct / total
                self._progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f}'.format(test_loss / (batch_idx + 1)))

    def train_model(self, epochs):
        print(f"{self.save_model_name} is training for {epochs} epochs.")
        train_loader, test_loader = _get_train_test_loaders(self.batch_size)

        for epoch in range(epochs):
            self.model.to(self.device)
            self.model.train()
            train_loss = 0
            correct, total = 0, 0
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predictions = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()
                self.train_acc = 100. * correct / total
                self._progress_bar(batch_idx, len(train_loader),
                                   'Epoch: {} | Loss: {:.3f}'.format(epoch + 1, train_loss / (batch_idx + 1)))

            self._test_model(test_loader)
            self.scheduler.step()
        print()
        return self.model, train_loader, test_loader

    def save_model(self, save_path=None):
        if save_path:
            file_path = save_path
        else:
            file_path = os.getcwd() + '/cifar10/' + self.save_model_name + '.pth'
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        model, model_name = _get_model_architecture(self.model_name, self.layer_number)
        model.load_state_dict(torch.load(file_path + model_name + '.pth'))
        model.eval()

        return model
