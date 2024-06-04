# TeamBananaBread
Machine learning class project at Unitn.

## Project
The models are described in the models directory, each model is its own python file. Common functionality can be extracted to a separate file or to `models/__init__.py`. The two models we built is an architecture based on ResNet with multiple classifiers found in `models/multclassifiers1.py`, the other is SWIN with background suppression inspired by HERBS found in the `models/HERBS.py` file

The requirements.txt file contains all the dependencies of the project.

To install the dependencies in your virtual environment run the following commands from the root directory of the project:
```bash
    # if you don't already have a virtual environment
    python3 -m venv .venv
    # activate the environment
    ## linux
    source .venv/bin/activate
    ## windows
    .\.venv\Scripts\activate.bat
    # install project in editable mode
    pip install -r requirements.txt
```
For faster synchronization to the virtual machine, it may be a good idea to store the venv outside the project directory.

### Datasets
All data is stored in the `../data/` directory; for easier synchronization of the code it is outside the project directory.
The datasets module is responsible for downloading and loading the data, here each dataset used is defined in its own file.
All dataloaders are defined in `datasets/data_fetch.py`

### Training
The main training logic is defined in the `trainer` package. `trainer.__init__.py` contains the main training and test loops.
For each model there is a specific file for its training, this is the entrypoint of the code.
Models are saved in the `results` directory.

## Assignment: Fine-grained classification
Fine-grained classification in Neural Networks for Computer Vision
refers to the task of categorizing images into very specific and
detailed classes, such as different species of birds or types of
flowers. This type of classification requires the model to be highly
accurate and sensitive to subtle differences in visual features, as the
distinctions between classes are often quite subtle.

To achieve fine-grained classification, neural
networks need to be trained on a large
dataset containing examples of each specific
class. The model must then learn to
distinguish between these classes based on
fine details and features that may be difficult for
humans to differentiate.

Fine-grained classification often involves the
use of specialized techniques, such as
attention mechanisms, object localization,
and data augmentation, to improve the
model's performance and accuracy. These
techniques help the model to focus on specific
regions of the image that are most relevant to
the classification task and to learn from a
diverse set of examples to generalize better to
unseen data.

### Authors
Maria Starodubtseva
Francesco di Massimo
Ricardo Esquivel
Márton Hévizi
