
import pandas as pd
import time
import numpy as np

        

def show_results(param_range,r1,r2,r3,r4,r5,no_rename=True):
    if no_rename:
    
        results = pd.DataFrame(data=np.column_stack((r1, r2, r3,r4,r5)), index=param_range,
                               columns=['Eu. Explicit', 'Eu. Implicit', 'Crank Nicolson','Monte Carlo', 'BS Analytic'])
        print(results)
    else:
        row = ['1 month', '2 months', '3 months', '6 months', '9 months', '1 year', '1.5 years', '2 years']
        results = pd.DataFrame(data=np.column_stack((r1, r2, r3,r4,r5)), index=row,
                               columns=['Eu. Explicit', 'Eu. Implicit', 'Crank Nicolson','Monte Carlo', 'BS Analytic'])
        print(results)
    
def show_time(t1,t2,t3,t4,t5):
    row_name = ['Time (s)']
    data = {
        'Eu. Explicit': [t1],
        'Eu. Implicit': [t2],
        'Crank Nicolson': [t3],
        'Monte carlo': [t4],
        'BS Analytic': [t5]
    }
    df = pd.DataFrame(data, index=row_name)
    print(df)

