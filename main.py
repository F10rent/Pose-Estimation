from dataset import get_dataloaders
from train import get_device
from dataset import get_transform
from dataset import get_datasets
from train import train_model
from model import CPM
from test import test_model
from test import get_predictions_ground_truths_heatmaps
from test import visualise_heapmat
from test import get_pck

def main():
    device = get_device()
    transform = get_transform()

    image_dir = "/lsp-master/images"
    joints_path = "/lsp-master/lsp-master/joints.mat"
    sigma = 1

    train_dataset, val_dataset, test_dataset = get_datasets(image_dir, joints_path, sigma, transform)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    model = CPM()
    train_model(model, train_loader, val_loader, device)

    test_model(model, test_loader, device)

    predictions, ground_truths, heatmaps = get_predictions_ground_truths_heatmaps(model, test_loader, device)
    visualise_heapmat(predictions, ground_truths, number=5)
    get_pck(predictions, ground_truths, heatmaps)


if __name__ == "__main__":
    main()