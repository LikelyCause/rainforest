"""Real datasets that ship with Rainforest, for trying it on real data next to the generated samples.

Each entry's Parquet file is built from its public source by `python -m app.sample_library.build` and committed under
data/. All four come from the UCI Machine Learning Repository under CC BY 4.0; `citation` is what to credit.
"""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

LIBRARY: dict[str, dict] = {
    "diabetes_readmission": {
        "name": "Diabetes readmission",
        "description": "Ten years of diabetic patient encounters at 130 US hospitals: demographics, diagnoses, "
                       "medications and lab results. Predict readmission (no, within 30 days, after 30 days). Heavy "
                       "missingness, ID columns and high-cardinality diagnosis codes.",
        "targets": ["readmitted"],
        "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008",
        "citation": "Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years "
                    "1999-2008 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J",
    },
    "student_grades": {
        "name": "Student grades",
        "description": "Portuguese-language students at two secondary schools: family, social and school "
                       "factors. Predict the first, second and final period grades (0-20) as a chain.",
        "targets": ["G1", "G2", "G3"],
        "source_url": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "citation": "Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C5TG7T",
    },
    "student_dropout": {
        "name": "Student dropout",
        "description": "Undergraduates at a Portuguese higher-education institution: enrolment, demographics, "
                       "economy and first-year results. Predict dropout, still enrolled or graduate (imbalanced). "
                       "Many categoricals are stored as numeric codes.",
        "targets": ["Target"],
        "source_url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
        "citation": "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' "
                    "Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C5MC89",
    },
    "steel_faults": {
        "name": "Steel plate faults",
        "description": "Geometry and luminosity measurements of faults on steel plates. Seven yes/no fault types "
                       "as targets, to try chains with many links.",
        "targets": ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"],
        "source_url": "https://archive.ics.uci.edu/dataset/198/steel+plates+faults",
        "citation": "Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. UCI Machine "
                    "Learning Repository. https://doi.org/10.24432/C5J88N",
    },
}


def path_of(key: str) -> Path:
    return DATA_DIR / f"{key}.parquet"
