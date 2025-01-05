from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from PIL import Image
from utilities.config_utils import TaskType
import matplotlib.pyplot as plt
import cv2
import numpy as np

class TacoDataset(Dataset):
    """∫
    Custom dataset for waste segmentation and classification using TACO dataset in COCO format
    
    params:
    - annotations_file: path to the annotations file
    - img_dir: path to the image directory
    - transforms: list of transformations to apply to the images

    returns:
    In case of segmentation task:
    - sample_img: image numpy array
    - masks: numpy array with masks for each object in the image
    In case of classification task:
    - sample_img: image numpy array
    - category_id: category id
    """


    def __init__(self, annotations_file: str, img_dir: str, transforms=None, task: TaskType=TaskType.SEGMENTATION) -> None:
        """ Constructor for the TacoDataset class """
        super().__init__()

        # Check if the provided paths are valid and if the task type is valid
        assert os.path.isfile(annotations_file), f"File not found: {annotations_file}"
        assert os.path.isdir(img_dir), f"Directory not found: {img_dir}"
        assert task in TaskType, f"Invalid task type: {task}"

        self.task = task
        self.coco_data = COCO(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = list(self.coco_data.imgs.keys())
        self.anns_ids = list(self.coco_data.anns.keys())
    

    def __len__(self) -> None:
        """ Returns the length of the dataset """

        if self.task == TaskType.SEGMENTATION:
            return len(self.img_ids)
        elif self.task == TaskType.CLASSIFICATION:
            return len(self.img_ids)
        else:
            raise NotImplementedError("Not a valid task type")
        
    
    def square_img(self, width: int, height: int, img: np.ndarray) -> np.ndarray:
        """
        Make the image square by adding padding to the image
        params:
        - width: width of the image
        - height: height of the image
        - img: image to make square
        returns:
        - img: square image
        """
        # Check if the image is already square
        if width == height:
            return img
        # If the image is not square, calculate the padding needed
        elif width > height:
            delta = width - height
            pad_top = delta // 2
            pad_bottom = delta - pad_top
            pad_left, pad_right = 0, 0
        else:  # height > width:
            delta = height - width
            pad_left = delta // 2
            pad_right = delta - pad_left
            pad_top, pad_bottom = 0, 0
            
        # Add padding to make the image square
        return cv2.copyMakeBorder(
            img,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
            )


    def __getitem__(self, idx) -> None:
        """ 
        Returns the sample and target (annotation) tensors at the given index 
        params:
        - idx: index of the image to retrieve
        returns:
        - sample: image tensor
        - target: annotation tensor
        """

        # Check if the task type is valid
        if self.task not in TaskType:
            raise NotImplementedError("Not a valid task type")
        
        # In case of segmentation task
        elif self.task == TaskType.SEGMENTATION:
            # Pick the image id based on given index
            img_id = self.img_ids[idx]
            # Load the image details using Coco API and image id
            img_coco_data = self.coco_data.loadImgs(img_id)[0] # The dict contains id, file_name, height, width, license and paths
            # Load path of the image using the image file name & Join the image directory path with the image file name
            path = os.path.join(self.img_dir, img_coco_data['file_name'])
            # Load the image using the path
            sample_img = Image.open(path)
            # Make the image a numpy array
            sample_img = np.array(sample_img)
            
            # Apply transformations to the image if they are provided
            if self.transforms:
                sample_img = self.transforms(sample_img)
            
            # Load the annotation for the image
            annotations = self.coco_data.loadAnns(self.coco_data.getAnnIds(imgIds=img_id))
            masks = [self.coco_data.annToMask(ann) for ann in annotations]
            masks = np.array(masks)

            # Apply transformations to the segmentations if they are provided
            if self.transforms:
                masks = self.transforms(masks)
            
            # Return the image in numpy array format and the masks in numpy array format
            return sample_img, masks
        
        # In case of classification task
        elif self.task == TaskType.CLASSIFICATION:

            # Pick the annotation id based on given index
            ann_id = self.anns_ids[idx]
            annotation = self.coco_data.loadAnns(ann_id)[0]
            bbox = annotation['bbox']  # it could be done using the bounding box instead of the segmentation
            category_id = annotation['category_id']
            # label = self.coco_data.loadCats(category_id)
            img_id = annotation['image_id']

            # Load the image details using Coco API and image id
            img_coco_data = self.coco_data.loadImgs(img_id)[0] # The dict contains id, file_name, height, width, license and paths
            # Load path of the image using the image file name & Join the image directory path with the image file name
            path = os.path.join(self.img_dir, img_coco_data['file_name'])
            # Load the image using the path
            sample_img = Image.open(path)
            # plt.imshow(sample_img)
            # plt.show()

            # Generate the mask from the annotation segmentation
            mask = self.coco_data.annToMask(annotation)
            # Make the image a numpy array to apply the mask
            sample_img = np.array(sample_img)
            # Apply the mask to the image
            sample_img = cv2.bitwise_and(sample_img, sample_img, mask=mask)
            # plt.imshow(sample_img)
            # plt.show()

            # Use bbox to crop the image
            x_min, y_min, width, height = [int(dim) for dim in bbox]
            cropped_image = sample_img[y_min:y_min+height, x_min:x_min+width]
            # plt.imshow(cropped_image)
            # plt.show()
            
            # Make the image square --> Make it is not necessary, but it could be a good practice
            sample_img = self.square_img(width, height, cropped_image)
            # plt.imshow(sample_img)
            # plt.show()

            # Apply transformations to the image if they are provided
            if self.transforms:
                sample_img = self.transforms(sample_img)
            
            # Return the image in numpy array format and the category id
            return sample_img, category_id

        # In case of segmentation task
        else:
            raise TypeError("This should never be reached error in TaskType")
        
        

taco_dataset = TacoDataset(annotations_file='data/train_annotations.json', img_dir='data', task=TaskType.SEGMENTATION)
print(taco_dataset[0])

    