# Simulator
This simulator serves as the training and evaluation platform in the following work:


> Efficient Large-Scale Fleet Management via Multi-Agent Deep Reinforcement Learning </br>
Kaixiang Lin, Renyu Zhao, Zhe Xu, Jiayu Zhou </br>
[KDD 2018 Long presentation](https://arxiv.org/abs/1802.06444)

### Prerequisites
- Python 3 and NumPy for the simulator (verified with Python 3.12 and 3.14).
- Python 3.10–3.12 and TensorFlow 2 for the optional learning algorithms.
  The algorithms retain the original session-based design through `tf.compat.v1`.

Install all project dependencies (simulator and learning algorithms) in a
Python 3.10–3.12 environment; Python 3.12 is recommended:

```bash
python -m pip install -r requirements.txt
```

If you only need the simulator, artificial-data example, or no-policy baseline,
install the lightweight dependencies instead:

```bash
python -m pip install -r requirements-simulator.txt
```

`requirements-algorithms.txt` remains an alias for the full dependencies.
The only directly required third-party packages are NumPy and TensorFlow.
Keras, TensorBoard and other TensorFlow dependencies are installed automatically.
Modules such as `pickle`, `pathlib`, `logging` and `unittest` are part of Python's
standard library and do not need separate installation.

### Run
```
python tests/run_example.py
```

This example uses artificial data and runs 144 simulation steps without a
learned dispatch policy. It can also be started from the `tests/` directory.

Run the compatibility checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Algorithm checks are skipped when TensorFlow is not installed.
All six checks were verified with Python 3.12 and TensorFlow 2.20.0.

### Training data and migration scope

The scripts in `run/` require the original external order and driver data.
These files are not included in this repository; update `dir_prefix` in each
script to your own data location before training. Legacy Python 2 pickle files
are read with `encoding='latin1'`, and results are written in binary mode.

This migration updates Python syntax, imports, integer indexing, serialization,
and TensorFlow compatibility. It does not revise the original reward design,
policy loss, or training/evaluation schedule. Passing the compatibility checks
does not establish reproduction of the paper's experimental results.
 
### Docs
Please find more details of usage in [Wiki](https://github.com/illidanlab/Simulator/wiki)


### References
If you find this work helpful in your research, please consider citing the following paper. The bibtex are listed below:
```
@article{lin2018efficient,
  title={Efficient Large-Scale Fleet Management via Multi-Agent Deep Reinforcement Learning},
  author={Lin, Kaixiang and Zhao, Renyu and Xu, Zhe and Zhou, Jiayu},
  journal={arXiv preprint arXiv:1802.06444},
  year={2018}
}
```
