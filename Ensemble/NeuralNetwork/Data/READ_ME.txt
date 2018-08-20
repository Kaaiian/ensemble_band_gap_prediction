
aflow folders:

aflow-data contains raw ".csv's" for the formula and properties

aflow-train contains the same ".csv's" with formula removed that matched fractional compositions to formula in the "Experimental_Band_Gap.csv".

aflow-jsons contains the citrination .PIF files and are parsed to get aflow-data.

mp folders follow the same logic.

------------------
need to do:
    get (and evaluate) various models for all the train data for both mp and aflow
    
    ensemble these models with the experimental models and compare performance