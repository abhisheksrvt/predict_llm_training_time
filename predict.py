import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

cost_per_gpu = 550 

X = np.array([25, 75, 320, 700, 1000, 2000, 3200]).reshape(-1, 1)
y = np.array([40, 75, 180, 330, 420, 780, 1260])

poly_model = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
poly_model.fit(X, y)

def cal_time(time):
    if (time <= 24):
        hrs = int(time)
        min = (time - hrs) * 60
        return 0, hrs, int(min)
    else:
        days = int(time/24)
        hrs = time - days * 24
        min = (hrs - int(hrs)) * 60
        return days, int(hrs), int(min)

try:
    print(f"\n<---------------Enter Parameters--------------->")
    input_model_size = float(input("\nEnter model size of parameters (in billion): "))
    input_tokens = float(input("Enter no of tokens to train (in billion): "))
    input_context_len = int(input("Enter context length: "))
    num_gpus = int(input("Enter no of gpus: "))
    predicted_time = poly_model.predict(np.array([[input_model_size * 1000]]))[0]
    predicted_time = predicted_time * input_tokens * min(1.3, 1 + input_context_len / 1e6)
    predicted_time /= 60
    total_time = predicted_time / num_gpus
    total_time_days, total_time_hrs, total_time_min = cal_time(total_time)

    print(f"\n<---------------Prediction Result--------------->")
    print(f"\nEstimated GPU-Hours: {int(predicted_time):,} hrs")
    print(f"Total Training Time: {total_time_days}d {total_time_hrs}h {total_time_min}m")
    total_cost = int(predicted_time * cost_per_gpu)
    if (total_cost > 100000 and total_cost < 10000000):
        print(f"Total Cost: ₹ {total_cost/100000:.2f} lakhs")
    elif (total_cost >= 10000000):
        print(f"Total Cost: ₹ {total_cost/10000000:.2f} cr")
    else:
        print(f"Total Cost: ₹ {total_cost:,}")
    print()
except ValueError:
    print("Invalid input. Please enter a valid value.")