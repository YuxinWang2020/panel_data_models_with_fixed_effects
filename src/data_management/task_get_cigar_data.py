"""
Compute the logarithm of all variables and adjust the monetary variables for the general
consumer price index. Also compute a lagged consumption variable.
"""
import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(SRC / "original_data" / "Cigar.csv")
@pytask.mark.produces(BLD / "data" / "cigar_clean.csv")
def task_get_cigar_data(depends_on, produces):
    cigar = pd.read_csv(depends_on, sep=",")

    # Data cleaning #
    cigar_clean = cigar.set_index(["state", "year"])

    cigar_clean["log_C_it"] = np.log(
        cigar_clean["sales"].mul(cigar_clean["pop"]).div(cigar_clean["pop16"])
    )
    cigar_clean["log_C_it_lag1"] = cigar_clean.groupby(["state"]).log_C_it.shift(
        1
    )  # Log C_{i,t-1}
    cigar_clean["log_P_it"] = np.log(cigar_clean["price"].div(cigar_clean["cpi"]))
    cigar_clean["log_Y_it"] = np.log(cigar_clean["ndi"].div(cigar_clean["cpi"]))
    cigar_clean["log_Pn_it"] = np.log(cigar_clean["pimin"].div(cigar_clean["cpi"]))

    cigar_clean = cigar_clean.dropna()  # remove NA

    cigar_clean.to_csv(produces, sep=",")
