import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

## Augmentation input variables - fine tuned for CIFAR10 Dataset
# horizontalFlipProb = 0.2
# shiftLimit = scaleLimit = 0.1
# shiftScaleRotateProb = 0.25
# maxHoles = minHoles = 1
# maxHeight = maxWidth = 16
# minHeight = minWidth = 16
# coarseDropoutProb = 0.5
# grayscaleProb = 0.15
# padHeightWidth = 40
# randomCropSize = 32
# randomCropProb = 1
# rotateLimit = 5


## Train and Teset Phase transformations
def albumentation_augmentation(mean, std):
    
    train_transforms = A.Compose([A.HorizontalFlip(p = horizontalFlipProb),
                                A.ShiftScaleRotate(shift_limit = shiftLimit, scale_limit = scaleLimit,
                                                   rotate_limit = rotateLimit, p = shiftScaleRotateProb),
                                A.CoarseDropout(max_holes = maxHoles, min_holes = minHoles, max_height = maxHeight,
                                                max_width = maxWidth, p = coarseDropoutProb, 
                                                fill_value = tuple([x * 255.0 for x in mean]),
                                                min_height = minHeight, min_width = minWidth),
                                A.ToGray(p = grayscaleProb),
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()
                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])
    
    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

## Train and Teset Phase transformations
def albumentation_augmentation_S8(mean, std):
    
    train_transforms = A.Compose([A.PadIfNeeded(min_height = padHeightWidth, min_width = padHeightWidth, always_apply = True),
                                A.RandomCrop(width = randomCropSize, height = randomCropSize, p = randomCropProb),
                                A.Rotate(limit = rotateLimit),
                                A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()
                              ])

    test_transforms = A.Compose([A.Normalize(mean = mean, std = std, always_apply = True),
                                ToTensorV2()])
    
    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]