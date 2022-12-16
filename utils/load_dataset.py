import torch
from utils.co3d_dataloader import CO3Dv2Wrapper
from utils.co3d_toy_dataloader import CO3Dv2ToyLoader

def load_dataset_test(args, image_size=None, masked=True):
    '''
    Load test dataset
    '''

    if args.dataset_name == 'co3d':
        
        if image_size is None:
            test_dataset = CO3Dv2Wrapper(root=args.root, category=args.category, sample_batch_size=32, subset='fewview_dev', stage='test', masked=masked)
        else:
            test_dataset = CO3Dv2Wrapper(root=args.root, category=args.category, sample_batch_size=32, subset='fewview_dev', stage='test', image_size=image_size, masked=masked)
    
    elif args.dataset_name == 'co3d_toy':
        test_dataset = CO3Dv2ToyLoader(root=args.root, category=args.category)

    else:
        raise NotImplementedError

    return test_dataset