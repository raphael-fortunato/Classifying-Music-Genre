from typing import BinaryIO, List

import bentoml
from PIL import Image
import torch

from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput, JsonOutput

CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
                         'jazz', 'metal', 'pop', 'reggae', 'rock']

@bentoml.env(pip_packages=['torch', 'numpy', 'torchvision'])
@bentoml.artifacts([PytorchModelArtifact('model')])
class GenreClassification(bentoml.BentoService):

    @bentoml.api(input=FileInput(), output=JsonOutput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        img_tensors = []
        for fs in file_streams:
            img_tensors.append(img)
        outputs = self.artifacts.classifier(torch.stack(img_tensors))
        _, output_classes = outputs.max(dim=1)
        import pdb; pdb.set_trace()
        
        return [CLASSES[output_class] for output_class in output_classes]
