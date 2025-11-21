# ReadMe

## Medical Information Retrieval

---

### What it does

---

Our software for medical information retrieval takes CT images of different body regions, analyses them and then finds similar images based on a query. 

This happens by extracting certain features of the images, including (but not limited to) a histogram, spatial features and thumbnail features. 

These features are then compared by different measures, for example the Manhattan distance or the cosine similarity. 

Based on the similarities the algorithm determines which other images in the same dataset are most similar to the chosen image (e.g. which other pictures show the CT image of a spine). 

Lastly, the algorithm gets evaluated. This evaluation is based on the IRMA code of the images. Each image is labelled with such a code and if the codes match, the images show the same body region and therefore have a high similarity. 

There are different approaches included for the evaluation, such as precision-at-k and the mean average precision. 

---

### Getting started

In order to run the program you need a python interpreter, as well as all of the following files (included in .zip-folder):

- evaluation.py
- feature_extractor.py
- main.py
- preprocessing.py
- query.py
- searcher.py
- codes.csv
- ImageCLEFmed2007_test

Run the evaluation file as this one accesses all of the other files and will include their important points. 

It may be necessary to change the image path if the program is not operating correctly (see Known Issues). 

---

### Known Issues

If you download the image set separately from the zip folder it can happen that the image path may not work correctly. 

This is a relatively easy fix as the debugger will tell you exactly which path in which file is causing issues and action can be taken accordingly. 
Please be extra wary of folders within folders as those have caused the most issues within our team. 

---

### Contributors

- Sophia Backert
- Annika Braun
- David Dietrich
- Elham Mohammadi
