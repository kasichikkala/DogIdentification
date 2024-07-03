import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import MyDataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from ResNet import resnet50



def test_model(model, test_loader, device):
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct_top1 += predicted[:, 0].eq(labels).sum().item()
            correct_top5 += predicted.eq(labels.view(-1, 1).expand_as(predicted)).sum().item()

            all_predictions.extend(predicted[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total

    print(f'Top-1 accuracy: {top1_accuracy:.2f}%')
    print(f'Top-5 accuracy: {top5_accuracy:.2f}%')


    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(20, 15))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, annot_kws={"size": 5})
    plt.xlabel('prediction')
    plt.ylabel('groundtruth label')
    plt.title('confusion matrix')


    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=4, rotation=45)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=4, rotation=0)
    plt.show()


    # errors = all_predictions != all_labels
    # most_confused_classes = np.unique(all_labels[errors], return_counts=True)
    # most_confused_pair_indices = np.unravel_index(np.argmax(most_confused_classes[1]), most_confused_classes[1].shape)
    # most_confused_true_label = most_confused_classes[0][most_confused_pair_indices[0]]
    # most_confused_predicted_label = most_confused_classes[0][most_confused_pair_indices[1]]
    #

    # cls_errors = (all_labels == most_confused_true_label) & (all_predictions == most_confused_predicted_label)
    # cls_errors_indices = np.where(cls_errors)[0]
    #

    # for i in range(2):
    #     img_path, true_label = test_dataset.data[cls_errors_indices[i]]
    #     predicted_label = all_predictions[cls_errors_indices[i]]

    #     img = Image.open(img_path)
    #     plt.subplot(1, 2, i + 1)
    #     plt.imshow(img)
    #     plt.title(f'True: {test_dataset.classes[true_label]}\nPredicted: {test_dataset.classes[predicted_label]}')
    #     plt.axis('off')
    #
    # plt.show()


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    batch_size = 128
    transform_val = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        # transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = MyDataset(root_dir=r"C:\Users\11312\PycharmProjects\dog_classification\images\test", transform=transform_val)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)


    model_path = "saved_models/resnet50.pth"
    model = resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 120)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    test_model(model, test_loader, device)
