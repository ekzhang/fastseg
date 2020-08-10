"""The `BaseSegmentation` class provides useful convenience functions for inference."""

import torch
import torch.nn as nn
from torchvision import transforms

# TODO(ekzhang): move to hashed weights from GitHub releases
MODEL_WEIGHTS_URL = {
    'mobilev3large-lraspp': 'https://www.dropbox.com/s/fgsv5bknwnn7mdj/mobilev3large-lraspp.pth?dl=1',
    'mobilev3small-lraspp': 'https://www.dropbox.com/s/rf19yi0svmwu0z5/mobilev3small-lraspp.pth?dl=1',
}

class BaseSegmentation(nn.Module):
    """Module subclass providing useful convenience functions for inference."""

    @classmethod
    def from_pretrained(cls, filename=None, **kwargs):
        """Load a pretrained model from a .pth checkpoint given by `filename`."""
        if filename is None:
            # Pull default pretrained model from internet
            name = cls.model_name
            if name in MODEL_WEIGHTS_URL:
                weights_url = MODEL_WEIGHTS_URL[name]
                checkpoint = torch.hub.load_state_dict_from_url(weights_url, map_location='cpu')
            else:
                raise ValueError(f'pretrained weights not found for model {name}, please specify a checkpoint')
        else:
            checkpoint = torch.load(filename, map_location='cpu')
        net = cls(checkpoint['num_classes'], **kwargs)
        net.load_checkpoint(checkpoint)
        return net

    def load_checkpoint(self, checkpoint):
        """Load weights given a checkpoint object from training."""
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = v
        self.load_state_dict(state_dict)

    def predict_one(self, image, return_prob=False, device=None):
        """Generate and return segmentation for a single image.

        See the documentation of the `predict()` function for more details. This function
        is a convenience wrapper that only returns predictions for a single image, rather
        than an entire batch.
        """
        return self.predict([image], return_prob, device)[0]

    def predict(self, images, return_prob=False, device=None):
        """Generate and return segmentations for a batch of images.

        Keyword arguments:
        images -- a list of PIL images or NumPy arrays to run segmentation on
        return_prob -- whether to return the output probabilities (default False)
        device -- the device to use when running evaluation, defaults to 'cuda' or 'cpu'
            (this must match the device that the model is currently on)

        Returns:
        if `return_prob == False`, a NumPy array of shape (len(images), height, width)
            containing the predicted classes
        if `return_prob == True`, a NumPy array of shape (len(images), num_classes, height, width)
            containing the log-probabilities of each class
        """
        # Determine the device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        # Preprocess images by normalizing and turning into `torch.tensor`s
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ipt = torch.stack([tfms(im) for im in images]).to(device)

        # Run inference
        with torch.no_grad():
            out = self.forward(ipt)

        # Return the output as a `np.ndarray` on the CPU
        if not return_prob:
            out = out.argmax(dim=1)
        return out.cpu().numpy()
