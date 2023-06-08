import random
import numpy as np
from torch.utils.data import Sampler
from mmdet.registry import DATA_SAMPLERS
from mmengine.fileio import get_local_path
from ..api_wrappers import COCO

COCOAPI = COCO

@DATA_SAMPLERS.register_module()
class CustomSampler(Sampler): # Every subclass of Sampler is required to have the __iter__ and the __len__ methods.
    def __init__(self,dataset,seed=None):
        self.seed = seed
        self.dataset = dataset
        print("Setting up the Class aware Custom Sampler")
        self.class_indices = self._get_class_indices()
        print("Total number of classes:", len(self.class_indices))
    def __iter__(self):
        # Return an iterable object which is basically
        # the indices that DataLoader will use.
        indices = self._generate_indices() # Generates ALL the indices which will be picked up by the DataLoader
        yield from indices  # We can also use return iter(indices) # Returning a Generator which can be used to iterate over the indices.

    def __len__(self):
        # Returns the length of the iterator above. In this case, it's the length of the dataset since 
        # we want to have only "len of training data_set" examples per epoch. (Even tho there might be duplicates)
        # to counter the class imbalance problem.
        return len(self.dataset)
        

    def _get_class_indices(self):
        num_classes = len(self.dataset.metainfo['classes'])
        class_indices = [[] for _ in range(num_classes)]
        self.dataset.load_data_list()
        # Iterate over all images
        for element in self.dataset.load_data_list():
            idx = element['img_id'] - 1 
            for instance in element['instances']:
                label = instance['bbox_label']
                if idx not in class_indices[label]:  
                    class_indices[label].append(idx)
                    
        return class_indices
    
    def _generate_indices(self):
        indices = []
        weights = [10 for i in range(len(self.class_indices))] # Giving a weight of 10 to all classes.
        # Giving specific superclasses thrice as much likelihood as others to be sampled.
        weights[0] = 30
        weights[6] = 30
        weights[9] = 30
        weights[10] = 30
        weights[12] = 30
        weights[21] = 30
        weights[22] = 30

        for _ in range(len(self.dataset)):
            class_idx = random.choices(range(len(self.class_indices)),weights = weights,k = 1)[0]
            sample_idx = np.random.choice(self.class_indices[class_idx])
            indices.append(sample_idx)
        return indices
        
