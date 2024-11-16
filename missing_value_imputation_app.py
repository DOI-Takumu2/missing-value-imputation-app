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
<div style="background-color: #dfffdf; padding: 10px; border-radius: 5px; line-height: 1.8;">
<b>このアプリは以下の方法でCSVファイルの欠損値を補完します：</b><br>
1. 段階的な平均値で補完<br>
2. ベイズ統計を用いて補完<br>
3. 回帰分析を用いて補完
</div>
""", unsafe_allow_html=True)

# 使用手順を黒線で囲む
st.markdown("""
<div style="border: 2px solid black; padding: 10px; margin: 10px; border-radius: 5px; line-height: 1.8;">
<b>### 使用手順:</b><br>
1. 欠損値を含むCSVファイルをアップロードしてください。<br>
2. 欠損値の補完方法を選択します。<br>
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
    df = df.copy()
    for col in df.columns:
        for i in range(len(df[col])):
            if pd.isnull(df[col][i]):  # 欠損値の場合
                prev_val = None
                next_val = None
                # 前の値を探す
                for j in range(i - 1, -1, -1):
                    if not pd.isnull(df[col][j]):
                        prev_val = df[col][j]
                        break
                # 次の値を探す
                for j in range(i + 1, len(df[col])):
                    if not pd.isnull(df[col][j]):
                        next_val = df[col][j]
                        break
                # 段階的な補完
                if prev_val is not None and next_val is not None:
                    gap = (j - i + 1)
                    step_avg = (next_val - prev_val) / gap
                    df[col][i] = prev_val + step_avg
                elif prev_val is not None:  # 次の値がない場合
                    df[col][i] = prev_val
                elif next_val is not None:  # 前の値がない場合
                    df[col][i] = next_val
    return df

def bayesian_fill(df):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    filled_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return filled_df

def regression_fill(df):
    filled_df = df.copy()
    for col in df.columns:
        if filled_df[col].isnull().any():
            train_data = filled_df[filled_df[col].notnull()]
            test_data = filled_df[filled_df[col].isnull()]
            X_train = train_data.drop(columns=[col])
            y_train = train_data[col]
            X_test = test_data.drop(columns=[col])

            # 特徴量がない場合はスキップ
            if X_train.shape[1] == 0:
                st.write(f"{col} の補完をスキップ（十分な特徴量がありません）")
                continue

            model = LinearRegression()
            model.fit(X_train, y_train)
            filled_df.loc[test_data.index, col] = model.predict(X_test)
    return filled_df

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

        # ダウンロード機能
        csv = filled_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 補完結果をダウンロード",
            data=csv,
            file_name="imputed_data.csv",
            mime="text/csv",
        )
