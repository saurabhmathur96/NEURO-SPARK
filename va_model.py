import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV

from data import read_data, discretize
from sklearn.metrics import log_loss
from scipy.stats import chi2

subset = [
     'SepsisMeningitisExcMyocarditis',
     'pH',
     'PreECLSArrest_Yes',
     'DeltapH',
     'Commoncarotidartery',
     'Hemolysis',
     'HCO324',
     'RenalReplacementTherapy',
     'RenalCr15_3',
     'MetaHyperbilirubinemia',
     'SaO2',
     'HCO3',
     'Bicarb_CO2change',
     'Epinephrine',
     'AgeDays',
     'RelativePCO2',
     'Pertussis',
     'PO224',
     'FiO2',
     'pH24',
     'Fio224',
     'milrinone',
     'CardioCPRrequired',
     'MetaGlucoseGT240',
     'PO2',
     'MechCircuitChange',
     'MetapHLT7_2',
     'Dobutamine',
     'NitricOxide',
     'SaO224',
     'PumpFlow4',
     'ProcintrapostOperative',
     'Vasopressin',
     'HTNrequiringvasodilators',
     'PulmonaryHypertension',
     'preECMOratio',
     'DeltaPO2',
     'relativeNormaMeanBPPerct',
     'normalizedMeanBP24'
][:19] + ['PrimaryComposite', 'Mode', 'relativeNormaMeanBPPerct']

frame = read_data(subset)
frame = frame[frame['Mode'] == 'VA']
frame.drop(['Mode'], axis=1, inplace=True)

bins = {
    'pH': [-np.inf, 7.0, 7.20, 7.35, 7.45, np.inf],
    'DeltapH': [-np.inf, -0.5, -0.1, 0, 0.1, 0.5, np.inf],
    'HCO324': [-np.inf, 10, 18, 24, 32, np.inf],
    'SaO2': [-np.inf, 70, 80, 88, 92, np.inf],
    'HCO3': [-np.inf, 10, 18, 24, 32, np.inf],
    'AgeDays': [-np.inf, 30, 6*30, 12*30, 5*12*30, 12*12*30, np.inf],   
    'RelativePCO2':[-np.inf, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, np.inf],
    'PO224': [-np.inf, 60, 110, 200, np.inf],
    'FiO2': [-np.inf, 20, 40, 60, np.inf],
    'pH24': [-np.inf, 7.0, 7.20, 7.35, 7.45, np.inf],
    'Fio224':  [-np.inf, 20, 40, 60, np.inf],
    'relativeNormaMeanBPPerct': [-np.inf, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, np.inf]
}

frame = discretize(frame, bins)

print (f"Read data set having {len(frame)} rows and {len(frame.columns)-1} features")

data, labels = frame.drop(['PrimaryComposite'], axis=1), frame.PrimaryComposite
data, data_test, labels, labels_test = train_test_split(data, labels, stratify=labels, test_size=0.2, random_state=0)
print (f"Split data into training set of size {len(data)} and test set of size {len(data_test)}")

class IntLogisticRegression(LogisticRegression):
    def __init__(self, factor, C):
        self.factor = factor
        super().__init__(C=C, 
                         max_iter=100, 
                         penalty="l1", 
                         solver="liblinear",
                         random_state=0,
                         fit_intercept=True)
        
    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)
        coef_ = np.round(self.coef_.ravel() * self.factor)[None, :]
        intercept_ = np.round(self.intercept_ * self.factor) 
        self.coef_ = coef_ / self.factor
        self.intercept_ = intercept_ / self.factor
        return self

a, b = 10, 17
clf = GridSearchCV(IntLogisticRegression(factor = a, C = 1), 
                   param_grid = dict(C = np.logspace(-4, 4, 10)),
                   scoring='neg_brier_score')
clf.fit(data, labels)
clf = clf.best_estimator_
print ("Fit logistic regression model")


variables = [v[:v.index("_")] if v.endswith("]") else v for v in data.columns.tolist()]
values = [v[v.index("_")+1:] if v.endswith("]") else 1 for v in data.columns.tolist()]
weights = pd.DataFrame(list(zip(variables, values, np.round(clf.coef_.ravel() * a).astype(int), np.round(np.exp(clf.coef_.ravel()), 4) )), 
                       columns = ['Variable', 'Value', 'Coefficient', 'Odds Ratio'])

print ("P(SNI = True | Features) = σ((Total Coefficients + Intercept - b)/a)")

print (weights[weights.Coefficient.abs() > 0.0][['Variable', 'Value', 'Coefficient', 'Odds Ratio']].to_string(index=False))
print (f"Intercept = {clf.intercept_.round(4).item()}")
print (f"a = {a}, b = {b}")
print ("AUC-ROC = %.4f" % roc_auc_score(labels_test, clf.predict_proba(data_test)[:, 1]))
print ("AUC-PR = %.4f" % average_precision_score(labels_test, clf.predict_proba(data_test)[:, 1]))


y_pred = clf.predict_proba(data*0)
y_true = labels
loglik0 = -log_loss(y_true, y_pred, normalize=False)

y_pred = clf.predict_proba(data)
loglik1 = -log_loss(y_true, y_pred, normalize=False)

G = -2*(loglik0 - loglik1)
p_value = chi2.sf(G, len(data.columns))
print (f"G = {np.round(G, 4)}, p = {p_value}")
