Provided here is the code to run NADE on the datasets used in

The Neural Autoregressive Distribution Estimator
Hugo Larochelle and Iain Murray
AISTATS 2011

This code requires Python and the Numpy library, including their
development header and library files (e.g., python-dev and
python-numpy-dev packages in Debian/Ubuntu). I've used Python 2.5.4
and Numpy 1.3.0, but the code will probably work with other similar
versions.

To use this code, simply follow these steps:

- Compile the C extensions by running the Makefile in mlpython/mathutils.

- Download the datasets by running "python download_datasets.py".

- Use run_nade.py to run NADE on the desired dataset, with the 
  chosen hyper-parameters.
  (example: "python run_nade.py nips 0.005 0 500 1234 1234 1")

File run_nade.py creates result files "results_DATASET_nade.txt" where
each line, corresponding to the result of a run, specifies the
hyper-parameters used and the results on the training, validation and
test sets. This allows someone to have several runs in parallel with
different hyper-parameters: the result file will then collect all the
results.

If you wish to apply NADE on some other dataset, I suggest you look at
the dataset modules in mlpython/datasets, to get an idea of how data
must be loaded before being fed to NADE. Then, write a new module for
your dataset (this should be easy if your dataset is in the LIBSVM
format). Finally, change run_nade.py so that it now allows you to load
your new dataset module (i.e. by adding your dataset to the string
list variable 'datasets'). By default, run_nade.py assumes that
examples are (input,target) pairs and removes the target before
training NADE (run_nade.py does unsupervised learning).  If your
dataset only has inputs without targets, look at datasets 'nips' and
'binarized_mnist' for how to proceed in this special case.

For questions and comments: Hugo Larochelle (larocheh@cs.toronto.edu).

Enjoy!
