# From NLP Zero to Language Model Hero: A Workshop on Artificial Intelligence

This repository accompanies the 2024 Equinor Developer Conference (EDC) workshop "From NLP Zero to Language Model Hero: A Workshop on Artificial Intelligence".

The workshop provides an introduction to Artificial Intelligence (AI), and more specifically large language models (LLMs). We will touch on why AI has been so successful in recent years and how it can be used for natural language processing.

## Outline

<table>
    <thead>
        <tr>
            <th>Title</th>
            <th>Type of work</th>
            <th>Material</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Course intro & setting up</td>
            <td></td>
            <td>

[Getting started](#getting-started)

</td>
        </tr>
        <tr>
            <td>What is artificial intelligence?</td>
            <td>Group work & interactive seminar</td>
            <td>

[Notebook 1](1_introduction/notebook.ipynb)

</td>
        </tr>
        <tr>
            <td>Efficient training of neural networks</td>
            <td>Group work & interactive seminar</td>
            <td>

[Notebook 2](2_inductive_biases_and_symmetries/notebook.ipynb)

</td>
        </tr>
        <tr>
            <td>Natural language processing using neural networks</td>
            <td>Group work & interactive seminar</td>
            <td>

[Notebook 3](3_language_models/notebook.ipynb)

</td>
        </tr>
    </tbody>
</table>


## Getting started

### Creating a local copy of the repository

The repository contains a library with classes and functions that we use throughout the workshop. It also contains notebooks that will guide you through the course. So you can clone the repository to follow the workshop on your own machine.

**Prerequisites:**
We are assuming that you have installed a recent Python version and an IDE, such as VS Code, that will make working with Python scripts and Jupyter notebooks easier.

1. Start by cloning the repository. Open a terminal and enter
```[bash]
git clone git@github.com:equinor/kai-edc2024-workshop.git
```
This will make a local copy of the repository. You can now move into the directory with ``cd kai-edc2024-workshop``.

2. From inside the directory, install the dependencies of the repository using
```[bash]
pip install .
```

3. Make sure that everything is installed properly by running
```
python run-tests.py --unit
```

If all tests pass, i.e. the execution ends with ``OK``, you are good to go! 🚀



## License Overview

The files in this repository are under various licenses:

- All the code is covered by the [MIT Licence](licenses/MIT.txt), unless otherwise stated.
- The code and content in the [`3_language_models`](3_language_models) directory is covered by the [Apache License 2.0](3_language_models/LICENCE).
- All the content is covered by the [Creative Commons BY Attribution-ShareAlike 4.0](licenses/CC%20BY-SA%204.0.txt) license, unless stated otherwise.