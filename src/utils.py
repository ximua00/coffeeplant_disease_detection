import os    

def normalization_parameters(dataloader):
    mean = 0.
    std = 0.
    for images, _ in dataloader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return mean, std

def make_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path