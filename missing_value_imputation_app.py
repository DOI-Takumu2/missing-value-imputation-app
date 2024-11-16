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
<b>使用手順:</b><br>
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

    # IterativeImputerで全体を一時的に補完
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    temp_df = pd.DataFrame(imputer.fit_transform(filled_df), columns=filled_df.columns)

    for col in df.columns:
        if filled_df[col].isnull().any():
            train_data = temp_df[filled_df[col].notnull()]
            test_data = temp_df[filled_df[col].isnull()]
            X_train = train_data.drop(columns=[col])
            y_train = train_data[col]
            X_test = test_data.drop(columns=[col])

            # 条件を満たさない場合はスキップ
            if X_train.shape[1] == 0 or len(X_train) < 5:
                st.warning(f"列 '{col}' の補完をスキップしました（十分な説明変数やデータがありません）")
                continue

            missing_rate = filled_df[col].isnull().sum() / len(filled_df[col])
            if missing_rate > 0.5:
                st.warning(f"列 '{col}' の補完をスキップしました（欠損率が高すぎます）")
                continue

            # データの正規化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # モデルの学習と予測
            model = LinearRegression()
            try:
                model.fit(X_train, y_train)
                filled_df.loc[test_data.index, col] = model.predict(X_test)
            except Exception as e:
                st.error(f"列 '{col}' の補完中にエラーが発生しました: {str(e)}")
                continue
    return filled_df

# ファイルアップロード部分
uploaded_file = st.file_uploader("📂 CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    st.write("アップロードされたファイルを処理中...")
    df = pd.read_csv(uploaded_file)
    st.write("アップロードされたデータ:")
    st.dataframe(df)

    method = st.selectbox("補完方法を選択", ["段階的な平均値", "ベイズ統計", "回帰分析"])

    # ボタンが押されたら補完処理を実行
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

# 各補完方法の説明を個別に表示
st.markdown("### 各補完方法の説明")

with st.expander("1️⃣ 段階的な平均値補完とは？"):
    st.markdown("""
    段階的な平均値補完では、欠損値の前後にある既知の値を利用し、その間の補完値を段階的に計算して推定します。この方法はデータが時間的な順序や規則性を持つ場合に特に効果的です。

    **適用例:**
    - 時系列データ（売上、気温、株価など）の欠損値補完。
    - 一部が抜け落ちた測定データの修復。

    **注意点:**
    - 前後の値が大きく異なる場合、不自然な補完値になる可能性があります。
    - 欠損値が連続して発生する場合、補完精度が低下することがあります。
    """)

with st.expander("2️⃣ ベイズ統計を用いた欠損値補完とは？"):
    st.markdown("""
    ベイズ統計を用いた補完では、他の列（変数）の情報を活用し、データ全体の相関や分布に基づいて欠損値を推定します。この方法は単純な補完方法に比べて精度が高い一方で、欠損値が多すぎる場合やデータ分布が大きく偏っている場合には注意が必要です。

    **適用例:**
    - データ間に強い相関関係がある。
    - 平均値や中央値ではデータ構造を十分に反映できない場合。

    **注意点:**
    - データが正規分布に従わない場合、補完値が実際の値とずれる可能性があります。
    - 欠損値が多すぎる場合、推定が不安定になることがあります。
    """)

with st.expander("3️⃣ 回帰分析を用いた欠損値補完とは？"):
    st.markdown("""
    回帰分析を用いた補完では、欠損値を他の列（変数）との相関関係から予測します。この方法では、欠損値がある列を目的変数とし、他の列を説明変数としてモデルを構築します。そして、学習済みモデルを用いて欠損値を推定します。

    **適用例:**
    - 各列が互いに強い相関を持つデータセット。
    - 欠損値が他の列の情報から合理的に推測可能な場合（例: 体重を年齢や身長から予測）。

    **注意点:**
    - 説明変数が不足している場合、補完精度が低下する可能性があります。
    - データに外れ値がある場合、モデルがそれに引きずられ、補完値が異常になることがあります。
    """)
