import pandas as pd
dfs = []
for p in ["googledp_cpp/gdp_native_results.csv","opendp_rust/opendp_native_results.csv"]:
    try:
        df = pd.read_csv(p)
        dfs.append(df)
    except FileNotFoundError:
        pass
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("combined_laplace_results.csv", index=False)
print("Wrote combined_laplace_results.csv with", len(combined), "rows")
