# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
import numpy as np

# アプリの基本設定
st.title("🧩 欠損値処理アプリ")
st.markdown("""
**データ分析を円滑に進めるための欠損値補完ツール**  
<small style="font-size: 12px; color: gray;">Missing Value Imputation Tool for Data Analysis.</small>

---

作成者：**土居拓務（DOI, Takumu）**
""", unsafe_allow_html=True)

# 黄緑色の背景で方法を表示
st.markdown("""
<div style="background-color: #dfffdf; padding: 10px; border-radius: 5px;">
このアプリは以下の方法でCSVファイルの欠損値を補完します：
1. 段階的な平均値で補完
2. ベイズ統計を用いて補完
3. 回帰分析を用いて補完
</div>
""", unsafe_allow_html=True)

# 使用手順を黒線で囲む
st.markdown("""
<div style="border: 2px solid black; padding: 10px; margin: 10px; border-radius: 5px;">
### 使用手順:
1. 欠損値を含むCSVファイルをアップロードしてください。
2. 欠損値の補完方法を選択します。
3. 補完結果を確認し、必要に応じてダウンロードしてください。
</div>
""", unsafe_allow_html=True)

st.markdown("""
---

**引用**:  
DOI, Takumu (2024). Missing Value Imputation Tool for Data Analysis.
""", unsafe_allow_html=True)

# 補完処理関数
def stepwise_fill(df):
    # 段階的平均値処理
    pass

def bayesian_fill(df):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    filled_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return filled_df

def regression_fill(df):
    # 回帰分析による補完
    pass

# ファイルアップロード部分
uploaded_file = st.file_uploader("📂 CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    st.write("アップロードされたファイルを処理中...")
    df = pd.read_csv(uploaded_file)
    st.write("アップロードされたデータ:")
    st.dataframe(df)

    method = st.selectbox("補完方法を選択", ["段階的な平均値", "ベイズ統計", "回帰分析"])
    if st.button("補完を実行"):
        if method == "段階的な平均値":
            filled_df = stepwise_fill(df)
        elif method == "ベイズ統計":
            filled_df = bayesian_fill(df)
        elif method == "回帰分析":
            filled_df = regression_fill(df)
        
        st.write("補完結果:")
        st.dataframe(filled_df)
