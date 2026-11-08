# UCI Air Quality Classification Suite — M.Tech (AIML) Machine Learning Assignment 2

## a. Problem Statement
Urban air pollution monitoring stations rely on arrays of low-cost metal-oxide chemical sensors and meteorological instruments to track air quality continuously. However, reference-grade analyzer readings often experience sensor drift or downtime. The objective of this project is to design, evaluate, and deploy an end-to-end supervised Machine Learning classification pipeline using the **UCI Air Quality Dataset (`AirQualityUCI.xlsx`)** to classify whether ambient hourly air quality is in a **High Carbon Monoxide Pollution Alert state (`target = 1`, where true hourly CO concentration `CO(GT) > 2.0 mg/m³`)** or **Normal Air Quality state (`target = 0`, where `CO(GT) <= 2.0 mg/m³`)** based on 13 chemical sensor, meteorological, and temporal features. Five classification models—**Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors (KNN)**, **Gaussian Naive Bayes**, and **Support Vector Machine (SVM)**—are trained and compared across six standard evaluation metrics and deployed via an interactive Streamlit web application.

---

## b. Dataset Description
* **Dataset Source**: UCI Machine Learning Repository — *Air Quality Dataset (`AirQualityUCI.xlsx`)* collected from an air quality chemical multisensor device deployed in a significantly polluted urban area in Italy.
* **Instance Size**: `7,344` valid labeled hourly instances after removing rows with missing ground-truth `CO(GT)` values (`-200` sentinel code) and sensor dropouts (well above the minimum requirement of `500` instances).
* **Feature Size**: `13` input features (`11` chemical/meteorological sensor attributes + `2` engineered temporal features, exceeding the minimum requirement of `12` features):
  1. `PT08.S1(CO)`: Tin oxide sensor response (nominally CO targeted)
  2. `C6H6(GT)`: True hourly averaged Benzene concentration ($\mu\text{g/m}^3$)
  3. `PT08.S2(NMHC)`: Titania sensor response (nominally NMHC targeted)
  4. `NOx(GT)`: True hourly averaged $\text{NO}_x$ concentration (ppb)
  5. `PT08.S3(NOx)`: Tungsten oxide sensor response (nominally $\text{NO}_x$ targeted)
  6. `NO2(GT)`: True hourly averaged $\text{NO}_2$ concentration ($\mu\text{g/m}^3$)
  7. `PT08.S4(NO2)`: Tungsten oxide sensor response (nominally $\text{NO}_2$ targeted)
  8. `PT08.S5(O3)`: Indium oxide sensor response (nominally $\text{O}_3$ targeted)
  9. `T`: Ambient Temperature ($^\circ\text{C}$)
  10. `RH`: Relative Humidity ($\%$)
  11. `AH`: Absolute Humidity
  12. `Month`: Calendar month extracted from `Date` (`1` to `12`)
  13. `Hour`: Hour of the day extracted from `Time` (`0` to `23`)
* **Target Variable (`target`)**: Binary label derived from ground-truth `CO(GT)`:
  * `0` (**Normal Air Quality**): $\text{CO(GT)} \le 2.0\text{ mg/m}^3$ (`4,261` instances, `58.02%`)
  * `1` (**High CO Pollution Alert**): $\text{CO(GT)} > 2.0\text{ mg/m}^3$ (`3,083` instances, `41.98%`)
* **Preprocessing Pipeline**: Replaced UCI sentinel missing values (`-200`) with `NaN`, performed an 80/20 stratified train-test split (`5,875` train rows / `1,469` holdout test rows in `test_data.csv`), applied median imputation (`SimpleImputer`) and z-score standardization (`StandardScaler`) fitted strictly on the training set.

---

## c. Mandatory Project Links
* **GitHub Repository Link**: `https://github.com/2026aj05121-cyber/2026AJ05121-ML-assignment-2`
* **Live Streamlit App Link**: `https://<your-app-name>.streamlit.app`

---

## d. Models Used & Evaluation Metrics Comparison Table

All 5 classification models were trained on the same scaled training set (`5,875` instances) and evaluated on the unseen holdout test dataset (`1,469` instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9163 | 0.9718 | 0.9038 | 0.8963 | 0.9000 | 0.8280 |
| Decision Tree | 0.9129 | 0.9584 | 0.8968 | 0.8963 | 0.8965 | 0.8211 |
| KNN | 0.9278 | 0.9765 | 0.9145 | 0.9141 | 0.9143 | 0.8519 |
| Naive Bayes | 0.8897 | 0.9542 | 0.8427 | 0.9060 | 0.8732 | 0.7769 |
| Support Vector Machine | 0.9346 | 0.9814 | 0.9237 | 0.9206 | 0.9221 | 0.8659 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved strong baseline performance (`Accuracy: 0.9163`, `AUC: 0.9718`, `MCC: 0.8280`). Because `PT08.S1(CO)` and `C6H6(GT)` share a strong monotonic relationship with carbon monoxide levels, a linear hyperplane captures the primary pollution trend well, though it slightly underfits non-linear humidity/temperature cross-sensitivities. |
| **Decision Tree** | Delivered competitive interpretability and solid metrics (`Accuracy: 0.9129`, `AUC: 0.9584`, `MCC: 0.8211`) with `max_depth=8`. Axis-aligned splits on `PT08.S1(CO)`, `NOx(GT)`, and `Hour` capture rush-hour traffic spikes, but orthogonal boundaries create step-function approximations on continuous chemical gradients. |
| **KNN** | Ranked second overall (`Accuracy: 0.9278`, `AUC: 0.9765`, `F1: 0.9143`, `MCC: 0.8519`). Distance-weighted voting ($k=9$) in standardized 13-dimensional sensor space effectively identifies local atmospheric regimes (e.g., cold winter morning inversion layers with high CO). |
| **Naive Bayes** | Recorded the lowest overall accuracy (`0.8897`) and MCC (`0.7769`), despite a high recall (`0.9060`) and AUC (`0.9542`). Metal-oxide sensors (`PT08.S1`, `PT08.S2`, `PT08.S5`) and benzene (`C6H6`) are heavily correlated during traffic emissions; this violates the Gaussian Naive Bayes conditional independence assumption and inflates false-positive alerts (`Precision: 0.8427`). |
| **Support Vector Machine** | Achieved the highest performance across every single evaluation metric (`Accuracy: 0.9346`, `AUC: 0.9814`, `Precision: 0.9237`, `Recall: 0.9206`, `F1: 0.9221`, `MCC: 0.8659`). The RBF kernel maps the 13 correlated chemical and meteorological features into a higher-dimensional space where non-linear sensor cross-sensitivities to temperature (`T`) and humidity (`RH`, `AH`) are cleanly separated. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the clear overall winner (`Accuracy: 93.46%`, `AUC: 0.9814`, `MCC: 0.8659`), outperforming all other classifiers by modeling non-linear electrochemical sensor interactions while maintaining balanced Precision and Recall. |