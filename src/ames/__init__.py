"""Ames Housing -> Real-Estate Valuation & Investment Risk Platform.

Five modules, each a layer of the stack:

    data      download + assemble raw sources into a single analysis frame
    features  ColumnTransformer, ordinal ladders, geo features
    avm       AVM accuracy metrics (MdAPE / PPE / FSD) + split-conformal intervals
    finance   amortisation, NOI, cap rate, DSCR, IRR, Monte Carlo
    risk      LTV, HPI stress paths, PD x LGD x EAD expected loss
"""

__version__ = "1.0.0"
