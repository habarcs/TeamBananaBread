# TeamBananaBread
Machine learning class project at Unitn.

## Project
The models are in the models directory, each model is its own python file. Common functionality can be extracted to a seperate file or to `models/__init__.py`

The data directory contains all the training and testing images.

The requirements.txt file contains all the dependencies of the project, if you add a new dependency please update it. (For now it only includes the cpu version of pytorch.)

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

### Code quality
Before commiting always format code with an autoformatter.
In pycharm  Ctrl+Alt+L on Windows and Linux or ⌘⌥L on macOS.

Also check for lint errors with pylint or the built-in linter.
On github after every commit, pylint is run, try to fix all errors before merging.

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

# Research papers
- [Fine grained classification survey](https://arxiv.org/pdf/2111.06119)
- [Awesome Fine-Grained Image Analysis](http://www.weixiushen.com/project/Awesome_FGIA/Awesome_FGIA.html)
-
