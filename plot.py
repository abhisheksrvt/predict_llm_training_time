import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.array([25, 75, 320, 700, 1000, 2000, 3200]).reshape(-1, 1)
y = np.array([40, 75, 180, 330, 420, 780, 1260])

poly_model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
poly_model.fit(X, y)

X_test = np.linspace(0, 3200, 1260).reshape(-1, 1)
y_pred_poly = poly_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred_poly, color='red', label='Polynomial Regression (deg=1)')
plt.xlabel('Model Parameters (Million)')
plt.ylabel('Training Time (Minutes)')
plt.title('Predicting Training Time from Model Size')
plt.legend()
plt.grid(True)

info = (
    "GPU: NVIDIA H200\n"
    "Tokens: 1 Billion\n"
    "Context Length: 2048\n"
    "Grad Accum Step: 2"
)
plt.text(0.95, 0.05, info, transform=plt.gca().transAxes,
         fontsize=10, va='bottom', ha='right',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()
