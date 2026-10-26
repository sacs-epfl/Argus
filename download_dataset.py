import torchvision

from decentralizepy.datasets.TinyImageNet import download_tiny_imagenet

if __name__ == "__main__":
    torchvision.datasets.CIFAR10(root="./eval/data/", train=True, download=True)
    torchvision.datasets.CIFAR10(root="./eval/data/", train=False, download=True)

    download_tiny_imagenet(root="./eval/data/")
