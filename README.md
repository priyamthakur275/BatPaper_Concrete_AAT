# Hybrid vs Non-Hybrid Soft Computing: Concrete Compressive Strength

Team: Priyam Thakur (1BM23CS252), Rajinder Kumar (1BM23CS260), Samir Chaudhary (1BM23CS294), Sana Subodh (1BM23CS296)
Guide: Dr. Roopashree S., B.M.S. College of Engineering (2026-27)

Sources
- Base paper: Aalimahmoody et al. (2021), Infrastructures 6(6), 80, https://www.mdpi.com/2412-3811/6/6/80 (DOI 10.3390/infrastructures6060080)
- Supporting reference: Xiao et al. (2025), Frontiers in Materials 12, 1698248, https://doi.org/10.3389/fmats.2025.1698248
- Dataset: UCI Concrete Compressive Strength, https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength (DOI 10.24432/C5PK67, CC BY 4.0; donor I-Cheng Yeh)
- data.csv is the UCI Concrete_Data.xls converted to CSV (1030 x 9), unmodified.

Models: MLR, ANN (8-8-1), GA-ANN, ANFIS. Five random 80/20 splits.

Run
    python3.12 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python show.py            # clean tables (best for presenting)
    python compare.py         # full table + verdict
    python demo.py            # ANN vs ANFIS
    python demo.py live       # trains both now (1 to 4 minutes)
    for s in 0 1 2 3 4; do python models.py $s; done   # re-run everything

Results from our runs are in the saved_results folder.


## Final model: ANFIS ensemble (hybrid)
`run_ens.py` trains 12 ANFIS models (24 to 72 rules) and averages them. Mean of 5 random 80/20 splits:
R2 0.918, RMSE 4.69 MPa, MAE 3.06 MPa. Base paper best model (BAT-ANN, 70/30 split): R2 0.913, RMSE 5.26, MAE 3.85.
Run `python show.py` to see all tables (saved results are in saved_results/).
