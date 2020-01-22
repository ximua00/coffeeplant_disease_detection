from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import json
import os


DATA_DIR = "../data/"

class2idx = {"healthy": 0, "red_spider_mite": 1, "rust_level_1": 2,
                    "rust_level_2": 3, "rust_level_3": 4, "rust_level_4": 5}
idx2class = {0: "healthy", 1: "red_spider_mite", 2: "rust_level_1",
                    3: "rust_level_2", 4: "rust_level_3", 5: "rust_level_4"}


class Plant():
    def __init__(self, imageID, csv_file="RoCoLE-csv.csv", output_size=(224, 224)):
        self.imageID = imageID
        self.csv_file = csv_file
        self.output_size = output_size
        self.im = Image.open(os.path.join(DATA_DIR, "images", self.imageID))
        self.im_size = self.im.size
        self.down_im = self.im.resize(output_size)
        self.process_data()
        self.bbox = self.extract_bbox()

        del self.im

    def process_data(self):
        data = pd.read_csv(os.path.join(
            DATA_DIR, "annotations", self.csv_file))
        data.set_index("External ID", inplace=True)
        target_row = data.loc[self.imageID]

        target_label = data.loc[self.imageID]["Label"]
        target_label = json.loads(target_label)

        self.classification = target_label["classification"]
        self.polygon = [(coordinate["x"], coordinate["y"])
                        for coordinate in target_label["Leaf"][0]["geometry"]]
        self.down_polygon = self.downsize_figure(self.polygon)

        return

    def extract_bbox(self):
        max_x, min_x = (0, self.output_size[0])
        max_y, min_y = (0, self.output_size[0])
        for x, y in self.polygon:
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

        return self.downsize_figure([(min_x, min_y), (max_x, max_y)])

    def downsize_figure(self, figure):
        w, h = self.im_size
        w_ratio = self.output_size[0] / w
        h_ratio = self.output_size[1] / h
        down_figure = [(round(w_ratio*x), round(h_ratio*y)) for x, y in figure]

        return down_figure

    def plot_image(self):
        im = self.draw_image_label(self.down_im, self.classification)
        im = self.draw_polygon(im)
        im = self.draw_bbox(im)
        im.show()

    def draw_image_label(self, im, label):
        draw = ImageDraw.Draw(im)
        (x, y) = (50, 50)
        color = 'rgb(0, 0, 0)'
        draw.text((x, y), label, fill=color)
        return im

    def draw_polygon(self, im):
        draw = ImageDraw.Draw(im, mode="RGBA")
        color = 'rgba(0, 0, 0, 125)'
        draw.polygon(self.down_polygon, fill=color, outline=color)
        return im

    def draw_bbox(self, im):
        draw = ImageDraw.Draw(im, mode="RGBA")
        color = 'rgba(0, 0, 0, 125)'
        draw.rectangle(self.bbox, fill=color, outline=color)
        return im


class PlantDataset(Dataset):
    def __init__(self, transforms, csv_file="RoCoLE-csv.csv", output_size=(224, 224)):
        self.transforms = transforms
        self.csv_file = csv_file
        self.output_size = output_size
        self.initialise_data()
        self.class2idx = class2idx
        self.idx2class = idx2class

    def initialise_data(self):
        data = pd.read_csv(os.path.join(
            DATA_DIR, "annotations", self.csv_file))
        self.image_ids = data['External ID'].tolist()

    def __getitem__(self, idx):
        plant = Plant(self.image_ids[idx])
        image = self.transforms(plant.down_im)
        target = self.class2idx[plant.classification]

        return image, target

    def __len__(self):
        return len(self.image_ids)


def get_dataloaders(batch_size=16):
    # estimated with utils.normalization_parameters
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4131, 0.5085, 0.2980],
                             [0.2076, 0.2035, 0.1974])
    ])

    dataset = PlantDataset(data_transforms)
    n_data = len(dataset)
    data_split = {"train": int(0.8*n_data),
                  "eval": int(0.1*n_data), "test": int(0.1*n_data)}
    train_data, eval_data, test_data = random_split(
        dataset, (data_split["train"], data_split["eval"], data_split["test"]))

    dataloaders = {"train": DataLoader(train_data, batch_size=batch_size, num_workers=4),
                   "eval": DataLoader(eval_data, batch_size=batch_size, num_workers=4),
                   "test": DataLoader(eval_data, batch_size=batch_size,num_workers=4)}

    return dataloaders


if __name__ == "__main__":
    pass

