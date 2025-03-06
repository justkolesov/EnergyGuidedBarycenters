import papermill as pm

for SETUP in [0,1]:
    for DIM in [50, 100, 1000]:
        for RSI in [0,1,2,3,4]:
            pm.execute_notebook(
               '../EnergyGuidedBarycenters/stylegan2/notebooks/SingleCell.ipynb',
               '../papermill/SingleCell.ipynb',
                parameters=dict(SETUP=SETUP, DIM=DIM, EPS=0.1, RSI=RSI)
            )
            
 