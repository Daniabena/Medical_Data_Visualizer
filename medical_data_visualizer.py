import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
bmi = df["weight"] / ((df["height"] / 100) ** 2)
df["overweight"] = 0
df.loc[bmi > 25, "overweight"] = 1

# 3
df.loc[df["cholesterol"] == 1, "cholesterol"] = 0
df.loc[df["cholesterol"] > 1, "cholesterol"] = 1

df.loc[df["gluc"] == 1, "gluc"] = 0
df.loc[df["gluc"] > 1, "gluc"] = 1


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        frame=df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )

    # 6
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7
    plot = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    # 8
    fig = plot.fig

    # 9
    fig.savefig("catplot.png")
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()

    df_heat = df_heat[df_heat["ap_lo"] <= df_heat["ap_hi"]]

    height_low = df_heat["height"].quantile(0.025)
    height_high = df_heat["height"].quantile(0.975)

    weight_low = df_heat["weight"].quantile(0.025)
    weight_high = df_heat["weight"].quantile(0.975)

    df_heat = df_heat[
        (df_heat["height"] >= height_low) &
        (df_heat["height"] <= height_high) &
        (df_heat["weight"] >= weight_low) &
        (df_heat["weight"] <= weight_high)
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones(corr.shape, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # 16
    fig.savefig("heatmap.png")
    return fig
