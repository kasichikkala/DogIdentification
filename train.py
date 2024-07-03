import torch
from torch import nn, optim
import torchvision.transforms as transforms
from utils import MyDataset, visualize_batch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ResNet import resnet50
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from torchsummary import summary


######## Parameters  ##########
batch_size = 256
Learningrate = 3e-3
epochs = 100

dataset_path = r"C:\Users\11312\PycharmProjects\dog_classification\images\Images"
pretrained_weights = "pretrained_models/resnet50-0676ba61.pth"#imagenet 1K v1 version https://download.pytorch.org/models/resnet50-0676ba61.pth
model_saved_path = "saved_models/"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def main():
    transform_train = transforms.Compose([
                                      transforms.Resize((256, 256),interpolation = InterpolationMode.BICUBIC),
                                      # transforms.Resize(256),
                                      transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.3),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(224),
                                      transforms.RandomApply([transforms.RandomRotation(15)],p=0.3),
                                      transforms.RandomEqualize(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])
    transform_val = transforms.Compose([
                                      transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                                      # transforms.Resize(256),
                                      # transforms.RandomHorizontalFlip(),
                                      # transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])

    my_dataset = MyDataset(root_dir=dataset_path)


    img_paths, labels = zip(*my_dataset.data)


    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )

    train_dataset = MyDataset(root_dir=dataset_path, transform=transform_train , img_paths=train_img_paths, labels=train_labels)
    val_dataset = MyDataset(root_dir=dataset_path, transform=transform_val, img_paths=val_img_paths, labels=val_labels)
    val_num = len(val_dataset)



    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)



    visualize_batch(train_loader, train_dataset)

    model = resnet50(num_classes=1000, include_top=True)


    model.load_state_dict(torch.load(pretrained_weights, map_location=device))

    # for param in model.parameters():
    #     param.requires_grad = False

    exclude_list = ['layer4.2', 'fc']

    for name, param in model.named_parameters():
        if not any(excluded_layer in name for excluded_layer in exclude_list):
            param.requires_grad = False
        print(name)
    num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 120)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 120)
    )


    model.to(device)

    summary(model, input_size=[(3, 224, 224)], device="cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=Learningrate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_acc = 0.0


    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    lr_list = []

    for epoch in range(epochs):
        print('-' * 30, '\n', f'Epoch {epoch + 1}/{epochs}')


        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0


        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device)) #[64, 120]
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels.to(device)).sum().item()

            if (step + 1) % 10 == 0:
                print(f'Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


        train_accuracy = 100. * correct_train / total_train
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)


        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for data_test in val_loader:
                test_images, test_labels = data_test
                outputs = model(test_images.to(device))
                loss = criterion(outputs, test_labels.to(device))
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += test_labels.size(0)
                correct_val += predicted.eq(test_labels.to(device)).sum().item()


        val_accuracy = 100. * correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        scheduler.step()  # update lr
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        print(f'Epoch {epoch + 1}: training loss: {train_losses[-1]:.4f} - training accuracy: {train_accuracies[-1]:.2f}%')
        print(f'lr: {current_lr:.6f} - val loss: {val_losses[-1]:.4f} - val accuracy: {val_accuracies[-1]:.2f}%')


        if val_accuracy > best_acc:
            best_acc = val_accuracy
            savename = model_saved_path + 'resnet50.pth'
            torch.save(model.state_dict(), savename)


        if (epoch + 1) % 5 == 0:
            plt.plot(lr_list, label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.savefig(f'learning_rate_curve.png')
            plt.close()

            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('loss_curve.png')
            plt.close()


            plt.plot(train_accuracies, label='Training Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('accuracy_curve.png')
            plt.close()

if __name__ == '__main__':
    main()
