# surface-decomposition
Code and explanation for research project about surface decomposition using genetic algorithm.

This project aims to decompose earth surface data with basis function as bell-shaped function using genetic optimization algorithm.


## How to run
```
python decomposition.py --data_path <nc_file>
ex) python decomposition.py --data_path ./data/ETOPO1_Ice_g_gdal.nc
```

## Experiment results
Left: raw data / Right: decomposed data
<img width="80%" src="project_results/ES_test_1.png"/>
<img width="80%" src="project_results/ES_test_2.png"/>

## Methodology
Description of bell-shaped function, parameter, and decomposed surface function below.
<img width="80%" src="project_results/method_1.png"/>

Description of genetic optimization algorithm below.
<img width="80%" src="project_results/method_2.png"/>
