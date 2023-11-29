A suite of tools to analyze the resonant dynamics of Kepler TTV systems and compare them to basic population synthesis models of resonance capture.


Analysis of TTV systems requires matplotlib, numpy, pandas. Running the population synthesis models also requires rebound and reboundx.
Computing resonant angles requires the posteriors from [Hadden & Lithwick (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154....5H/abstract) (available [here](https://zenodo.org/record/162965)) 
and [Jontof-Hutter et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021AJ....161..246J/abstract) (available [here](https://zenodo.org/record/4422053)). 
Additionally, the [cumulative KOI table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative) in .csv format from the Exoplanet Archive is required.
These files should be placed in a top-level directory named `data`.

If you use this code, please cite [Goldberg & Batygin (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...948...12G/abstract).
