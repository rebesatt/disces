# DISCES: Systematic Discovery of Event Stream Queries

We propose four algorithms for the discovery of event stream queries given a stream database.

A detailed description of the algorithms can be found in the [DISCES technical report](./disces_technical%20report.pdf).

The experiments can be reproduced by running the scripts in the `experiments` directory. The results will be saved in the directory `experiment_results`.

In order to run the experiments, first download the repository with:
```
https://github.com/rebesatt/disces.git
```

Then inside the repository run the command:
```bash
python reproduce_paper.py
```
To see further options run:
```bash
python reproduce_paper.py -h
```