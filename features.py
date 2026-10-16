"""
Feature definitions shared by the web form (templates/index.html) and the server-side
validation in app.py. The numeric codes for categorical features follow the coding of
data/heart.csv (see README -> "Feature dictionary").
"""

# kind = "number": min / max / step / unit are used.  kind = "choice": options = [(code, label), ...]
FEATURE_GROUPS = [
    {
        "title": "About the patient",
        "fields": [
            {"name": "age", "label": "Age", "kind": "number", "min": 18, "max": 100,
             "step": 1, "unit": "years", "placeholder": "54"},
            {"name": "sex", "label": "Sex", "kind": "choice",
             "options": [(1, "Male"), (0, "Female")]},
        ],
    },
    {
        "title": "Measurements at rest",
        "fields": [
            {"name": "trestbps", "label": "Resting blood pressure", "kind": "number",
             "min": 80, "max": 250, "step": 1, "unit": "mm Hg", "placeholder": "130"},
            {"name": "chol", "label": "Serum cholesterol", "kind": "number",
             "min": 100, "max": 600, "step": 1, "unit": "mg/dl", "placeholder": "240"},
            {"name": "fbs", "label": "Fasting blood sugar above 120 mg/dl", "kind": "choice",
             "options": [(0, "No"), (1, "Yes")]},
            {"name": "restecg", "label": "Resting ECG result", "kind": "choice",
             "options": [(1, "Normal"), (0, "Left ventricular hypertrophy"),
                         (2, "ST-T wave abnormality")]},
        ],
    },
    {
        "title": "Exercise stress test",
        "fields": [
            {"name": "thalach", "label": "Maximum heart rate reached", "kind": "number",
             "min": 60, "max": 220, "step": 1, "unit": "bpm", "placeholder": "150"},
            {"name": "exang", "label": "Chest pain brought on by exercise", "kind": "choice",
             "options": [(0, "No"), (1, "Yes")]},
            {"name": "oldpeak", "label": "ST depression compared with rest", "kind": "number",
             "min": 0, "max": 7, "step": 0.1, "unit": "mm", "placeholder": "1.0"},
            {"name": "slope", "label": "Slope of the peak-exercise ST segment", "kind": "choice",
             "options": [(2, "Upsloping"), (1, "Flat"), (0, "Downsloping")]},
        ],
    },
    {
        "title": "Symptoms and imaging",
        "fields": [
            {"name": "cp", "label": "Chest pain type", "kind": "choice",
             "options": [(0, "No symptoms (asymptomatic)"), (1, "Atypical angina"),
                         (2, "Non-anginal pain"), (3, "Typical angina")]},
            {"name": "ca", "label": "Major vessels coloured by fluoroscopy", "kind": "choice",
             "options": [(0, "0"), (1, "1"), (2, "2"), (3, "3")]},
            {"name": "thal", "label": "Thallium stress test result", "kind": "choice",
             "options": [(2, "Normal"), (1, "Fixed defect"), (3, "Reversible defect")]},
        ],
    },
]

FEATURES = [f for g in FEATURE_GROUPS for f in g["fields"]]
FEATURES_BY_NAME = {f["name"]: f for f in FEATURES}

# Illustrative profiles for the "Try an example" buttons. These are made-up values,
# not real patients.
EXAMPLES = {
    "lower": {"age": 41, "sex": 0, "cp": 2, "trestbps": 120, "chol": 204, "fbs": 0,
              "restecg": 1, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 2,
              "ca": 0, "thal": 2},
    "higher": {"age": 63, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233, "fbs": 1,
               "restecg": 0, "thalach": 112, "exang": 1, "oldpeak": 2.6, "slope": 1,
               "ca": 2, "thal": 3},
}
