import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide"
)

# ===============================
# 2. LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("bank_churn_data.csv")

    # standardisasi kolom (ANTI ERROR)
    df.columns = df.columns.str.lower()

    # hapus kolom id jika ada
    drop_cols = ["clientnum", "user_id", "id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # encode target
    if df["attrition_flag"].dtype == "object":
        df["attrition_flag"] = df["attrition_flag"].map(
            {"Attrited Customer": 1, "Existing Customer": 0}
        )

    return df


df = load_data()

# ===============================
# 3. TITLE
# ===============================
st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Aplikasi untuk **analisis** dan **prediksi churn nasabah bank** "
    "menggunakan **Random Forest** dan **Streamlit**."
)

if df is None:
    st.stop()

# ===============================
# 4. SIDEBAR
# ===============================
st.sidebar.header("📌 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dataset Overview", "Exploratory Data Analysis (EDA)", "Prediction Model"]
)

st.sidebar.markdown("---")
st.sidebar.write("👤 Student Project")
st.sidebar.write("📘 Assignment EC9")

# ===============================
# 5. DATASET OVERVIEW
# ===============================
if menu == "Dataset Overview":
    st.header("📂 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nasabah", df.shape[0])
    col2.metric("Total Fitur", df.shape[1])
    col3.metric("Churn Rate", f"{df['attrition_flag'].mean()*100:.2f}%")

    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

# ===============================
# 6. EDA
# ===============================
elif menu == "Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")

    # filter
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.selectbox(
            "Filter Gender",
            ["All"] + sorted(df["gender"].unique())
        )
    with col2:
        income_filter = st.selectbox(
            "Filter Income",
            ["All"] + sorted(df["income_category"].unique())
        )

    df_plot = df.copy()
    if gender_filter != "All":
        df_plot = df_plot[df_plot["gender"] == gender_filter]
    if income_filter != "All":
        df_plot = df_plot[df_plot["income_category"] == income_filter]

    # 1. Pie chart churn
    st.subheader("1. Proporsi Churn Nasabah")
    churn_count = df_plot["attrition_flag"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(
        churn_count,
        labels=["Existing", "Churn"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff9999"]
    )
    st.pyplot(fig1)

    # 2. Boxplot umur
    st.subheader("2. Umur vs Churn")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        x="attrition_flag",
        y="customer_age",
        data=df_plot,
        ax=ax2
    )
    st.pyplot(fig2)

    # 3. Education distribution
    st.subheader("3. Distribusi Pendidikan")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.countplot(
        y="education_level",
        data=df_plot,
        order=df_plot["education_level"].value_counts().index,
        ax=ax3
    )
    st.pyplot(fig3)

    # 4. Education vs Income
    st.subheader("4. Pendidikan vs Pendapatan")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(
        x="education_level",
        hue="income_category",
        data=df_plot,
        ax=ax4
    )
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# ===============================
# 7. PREDICTION MODEL
# ===============================
elif menu == "Prediction Model":
    st.header("🤖 Prediksi Churn Nasabah")
    st.info("Masukkan data nasabah untuk memprediksi risiko churn.")

    df_model = df.copy()

    # encoding
    edu_map = {
        "Uneducated": 0, "High School": 1, "College": 2,
        "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5, "Unknown": -1
    }
    income_map = {
        "Less than $40K": 0, "$40K - $60K": 1,
        "$60K - $80K": 2, "$80K - $120K": 3,
        "$120K +": 4, "Unknown": -1
    }
    card_map = {"Blue": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

    df_model["education_level"] = df_model["education_level"].map(edu_map)
    df_model["income_category"] = df_model["income_category"].map(income_map)
    df_model["card_category"] = df_model["card_category"].map(card_map)

    le_gender = LabelEncoder()
    le_marital = LabelEncoder()

    df_model["gender"] = le_gender.fit_transform(df_model["gender"])
    df_model["marital_status"] = le_marital.fit_transform(df_model["marital_status"])

    df_model = df_model.fillna(0)

    X = df_model.drop("attrition_flag", axis=1)
    y = df_model["attrition_flag"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # feature importance
    st.subheader("🔍 Faktor Utama Penyebab Churn")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    fig_imp, ax_imp = plt.subplots()
    feat_imp.nlargest(10).plot(kind="barh", ax=ax_imp)
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    st.divider()

    # input
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Umur", 20, 80, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent = st.slider("Jumlah Tanggungan", 0, 5, 2)
        education = st.selectbox("Pendidikan", list(edu_map.keys()))
        marital = st.selectbox("Status Pernikahan", df["marital_status"].unique())
        income = st.selectbox("Pendapatan", list(income_map.keys()))

    with col2:
        card = st.selectbox("Jenis Kartu", list(card_map.keys()))
        months_book = st.number_input("Lama Jadi Nasabah (bulan)", 12, 60, 36)
        total_rel = st.number_input("Total Relasi Produk", 1, 6, 3)
        inactive_12 = st.number_input("Bulan Tidak Aktif (1 thn)", 0, 12, 1)
        contacts_12 = st.number_input("Kontak CS (1 thn)", 0, 10, 2)
        trans_amt = st.number_input("Total Transaksi", 0, 20000, 4000)
        trans_ct = st.number_input("Frekuensi Transaksi", 0, 150, 60)
        revolve_bal = st.number_input("Saldo Revolving", 0, 3000, 1000)

    if st.button("🔍 Prediksi"):
        input_data = X.mean().to_frame().T

        input_data["customer_age"] = age
        input_data["gender"] = le_gender.transform([gender])[0]
        input_data["dependent_count"] = dependent
        input_data["education_level"] = edu_map[education]
        input_data["marital_status"] = le_marital.transform([marital])[0]
        input_data["income_category"] = income_map[income]
        input_data["card_category"] = card_map[card]
        input_data["months_on_book"] = months_book
        input_data["total_relationship_count"] = total_rel
        input_data["months_inactive_12_mon"] = inactive_12
        input_data["contacts_count_12_mon"] = contacts_12
        input_data["total_trans_amt"] = trans_amt
        input_data["total_trans_ct"] = trans_ct
        input_data["total_revolving_bal"] = revolve_bal

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ Risiko CHURN tinggi ({prob*100:.1f}%)")
        else:
            st.success(f"✅ Nasabah diprediksi TETAP ({prob*100:.1f}% risiko churn)")
