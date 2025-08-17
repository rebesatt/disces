# DISCES: Systematic Discovery of Event Stream Queries

We propose four algorithms for the discovery of event stream queries given a stream database.

A detailed description of the algorithms can be found in the [DISCES technical report](./disces_technical%20report.pdf).

---

## Reproducing the Experiments

The experiments from the paper can be reproduced using the scripts in the `experiments` directory.  
Intermediate results will be stored in the directory `experiment_results`.  
The final PDF report will be written to the repository root as **`main.pdf`**.

### 1. Clone the repository
```
git clone https://github.com/rebesatt/disces.git
cd disces
```

### 2. Requirements

The script requires Python 3 and the following Python packages:

- requests  
- pandas  
- msgpack  
- numpy  
- matplotlib  
- seaborn  
- func_timeout  
- jinja2  
- typing_extensions  

Missing packages can be installed via:
```
python -m pip install -r requirements.txt
```

In addition, a working **LaTeX installation** is required to generate the final PDF (`main.pdf`).

### 3. Run the experiments

#### Quick run (several hours)
Runs all algorithms **except IL-Miner**.  
This mode is recommended for most users, as it typically finishes within a few hours.
```
python reproduce_paper.py
```

#### Full run (several days)
Runs the **complete experiment including IL-Miner**.  
⚠️ This mode can take **multiple days** to finish, depending on your hardware.
```
python reproduce_paper.py -ilm
```

### 4. Results

At the end of either run, the script generates:

- Intermediate data in `experiment_results/`  
- A final PDF report in the repository root:

  ```
  main.pdf
  ```

This report reproduces the figures and tables from the article and can be compared directly to the original.

### 5. Options

To see all available options, run:
```
python reproduce_paper.py -h
```