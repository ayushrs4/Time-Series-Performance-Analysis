import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
aapl = pd.read_csv("../data/AAPL.csv")
sp500 = pd.read_csv(
    "../data/SP500.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

# -----------------------------
# Preprocess
# -----------------------------
aapl["Date"] = pd.to_datetime(aapl["Date"])
sp500["Date"] = pd.to_datetime(sp500["Date"])

aapl = aapl.sort_values("Date")
sp500 = sp500.sort_values("Date")

aapl_price = "Adj Close" if "Adj Close" in aapl.columns else "Close"
sp500_price = "Adj Close" if "Adj Close" in sp500.columns else "Close"

aapl = aapl[["Date", aapl_price]].rename(columns={aapl_price: "AAPL"})
sp500 = sp500[["Date", sp500_price]].rename(columns={sp500_price: "SP500"})

df = pd.merge(aapl, sp500, on="Date", how="inner")

# -----------------------------
# Returns & Risk
# -----------------------------
df["AAPL_Return"] = df["AAPL"].pct_change()
df["SP500_Return"] = df["SP500"].pct_change()

df["AAPL_Cum"] = (1 + df["AAPL_Return"]).cumprod()
df["SP500_Cum"] = (1 + df["SP500_Return"]).cumprod()

# -----------------------------
# Metrics
# -----------------------------
metrics = {}
for asset in ["AAPL_Return", "SP500_Return"]:
    annual_return = df[asset].mean() * 252
    annual_vol = df[asset].std() * np.sqrt(252)
    sharpe = annual_return / annual_vol

    metrics[asset] = {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe
    }

print("Performance Metrics:")
for k, v in metrics.items():
    print(f"\n{k}")
    for metric, value in v.items():
        print(f"{metric}: {value:.4f}")

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.plot(df["Date"], df["AAPL_Cum"], label="Apple (AAPL)")
plt.plot(df["Date"], df["SP500_Cum"], label="S&P 500")
plt.title("Cumulative Returns: Apple vs S&P 500")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
import os

os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/apple_vs_sp500.png", dpi=300, bbox_inches="tight")
plt.show()