import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# 1. Download the cityscapes.py and store it in your project directory
# 2. From your other script you just need to import the dataloader and get_img_paths by doing || from cityscapes import Cityscapes,get_img_paths ||
# 3. Specify the input_directory as well as the target directory 
# (ex. 
# || input_dir = "../data/semantic-segmentation/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train"  ||
# || target_dir = ""../data/semantic-segmentation/cityscapes/gtFine_trainvaltest/gtFine/train") ||
# 4. Get the img_paths by doing: || input_img_paths,target_img_paths = get_img_paths(input_dir,target_dir) ||
# 5. From here you can define your training and validation splits (I have also created a way to do stratified splits but that is a topic of another video). To do this you can do the following:
# || 
#   val_samples = 200
#   random.Random(1337).shuffle(input_img_paths)
#   random.Random(1337).shuffle(target_img_paths)
#   train_input_img_paths = input_img_paths[:-val_samples]
#   train_target_img_paths = target_img_paths[:-val_samples]
#   val_input_img_paths = input_img_paths[-val_samples:]
#   val_target_img_paths = target_img_paths[-val_samples:]
# || 
# 6. Specify the other parameters like img_size (ex. || img_size = (160,160) || ),  batch_size (ex. || batch_size=16 ||)
# 7. Get your train_gen and val_gen: 
#  train_gen = Cityscapes(batch_size, img_size, train_input_img_paths, train_target_img_paths)
#  val_gen = Cityscapes(batch_size, img_size, val_input_img_paths, val_target_img_paths)
# 8. To train you then just create your TF/Keras model and do
# model.fit(self.train_gen,epochs=epochs, validation_data=self.val_gen, callbacks=callbacks, verbose=0)

# I hope this helps and sorry again! Happy coding!

#CHANGE THE PATHS!!!
input_dir = "cityscapes\\leftImg8bit\\train" #CHANGE THE PATHS!!!
target_dir = "cityscapes\\gtFine\\train" #CHANGE THE PATHS!!!
img_size = (512, 512)
num_classes = 24
batch_size = 32

def get_img_paths(input_dir, target_dir):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, city, fname) 
            for city in os.listdir(input_dir) 
            for fname in os.listdir(os.path.join(input_dir, city))
                if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, city, fname) 
            for city in os.listdir(target_dir) 
            for fname in os.listdir(os.path.join(target_dir, city))
                if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
        ]
    )
    return input_img_paths, target_img_paths


class Cityscapes(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.void_classes = [0, 1, 2, 3, 4, 5, 9, 10, 18, 29, 30, -1] #not to train
        self.valid_classes = [6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(num_classes)))

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y_temp = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (24,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y_temp[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, ..., 30. Subtract one to make them 0, 1, ..., 29:
            y_temp[j] -= 1
            y_temp[j] = self.fix_indxs(y_temp[j])
            y[j] = self.one_hot_encode(y_temp[j])
        return x, y
    
    def fix_indxs(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask == 255] = len(self.valid_classes)
        return mask
    
    def one_hot_encode(self,lbl):
        new_lbl = np.array(self.get_one_hot(lbl.reshape(-1),num_classes))
        new_lbl = new_lbl.reshape(512,512,num_classes)
        return new_lbl
        
    def get_one_hot(self,targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
