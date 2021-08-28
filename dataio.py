from torchvision import transforms
from torchvision.datasets import LSUN

def get_data():

    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

    print('Starting load lsun dataset!')

    dset = LSUN('D:\Datasets\LSUN', classes=['bedroom_train'], transform=transform)

    print('Finishing loading!')

    return dset