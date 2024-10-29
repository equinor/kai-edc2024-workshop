# From NLP Zero to Language Model Hero: A Workshop on Artificial Intelligence

This repository accompanies the 2024 Equinor Developer Conference (EDC) workshop "From NLP Zero to Language Model Hero: A Workshop on Artificial Intelligence".

The workshop provides an introduction to Artificial Intelligence (AI), and more specifically large language models (LLMs). We will touch on why AI has been so successful in recent years and how it can be used for natural language processing.

## Outline

<table>
    <thead>
        <tr>
            <th>Time</th>
            <th>Title</th>
            <th>Type of work</th>
            <th>Material</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>08:00 - 08:30</td>
            <td>Course intro & setting up</td>
            <td></td>
                        <td>

[Getting started](#getting-started)

</td>
        </tr>
        <tr>
            <td>08:30 - 09:15</td>
            <td>AI and natural language processing</td>
            <td>Interactive lecture</td>
            <td>



</td>
        </tr>
        <tr>
            <td>09:15 - 09:30</td>
            <td colspan=3>

**Coffee break**

</td>
        </tr>
        <tr>
            <td>09:30 - 10:45</td>
            <td>What is artificial intelligence?</td>
            <td>Group work & interactive seminar</td>
            <td>

[Notebook]([1_introduction/notebook.ipynb)

</td>
        </tr>
        <tr>
            <td>10:45 - 11:00</td>
            <td colspan=3>

**Coffee break**

</td>
        </tr>
        <tr>
            <td>11:00 - 12:00</td>
            <td>Efficient training of neural networks</td>
            <td>Group work & interactive seminar</td>
                        <td>

TODO

</td>
        </tr>
        <tr>
            <td>12:00 - 13:00</td>
            <td colspan=3>

**Lunch break**

</td>
        </tr>
        <tr>
            <td>13:00 - 14:30</td>
            <td>Natural language processing using neural networks</td>
            <td>Group work & interactive seminar</td>
            <td>

[Notebook]([3_language_models/notebook.ipynb)

</td>
        </tr>
        <tr>
            <td>14:30 - 14:45</td>
            <td colspan=3>

**Coffee break**

</td>
        </tr>
        <tr>
            <td>14:45 - 16:30</td>
            <td>Large language models</td>
            <td>Group work & interactive seminar</td>
            <td>
                
TODO
            
</td>
        </tr>
        <tr>
            <td>16:30 - 16:45</td>
            <td colspan=3>

**Coffee break**

</td>
        <tr>
            <td>16:45 - 17:00</td>
            <td>Wrap up</td>
            <td>Interactive lecture</td>
            <td></td>
        </tr>
    </tbody>
</table>


## Getting started

We have prepared two ways to get started with the workshop: Option 1. GitHub Codespace; and option 2: Creating a local copy of the repository.

### Option 1: GitHub Codespace

This is the simplest option to get started with the workshop and just involves scrolling up to the page and clicking on the green **Code** button. This will open a small pop-up window with a second green button with the name **Create codespace on main**. Click on that button and wait for the setup of the codespace to finish (this may take up to 5 minutes). You are now good to go! 🚀


### Option 2: Creating a local copy of the repository

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




 
