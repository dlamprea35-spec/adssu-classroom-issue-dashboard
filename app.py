import re
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="ADSSU Classroom Issue Dashboard", layout="wide")

# Keywords used to find which facility items are mentioned in a report
KEYWORDS = {
    "Aircon": ["aircon", "air conditioner"],
    "Ceiling Fan": ["ceiling fan", "electric fan", "wall fan"],
    "Exhaust Fan": ["exhaust fan"],
    "Light": ["fluorescent", "light bulb", "bulb", "light fixture", "lights"],
    "Outlet": ["outlet", "socket", "extension cord"],
    "Wiring/Switch": ["wiring", "switch", "breaker"],
    "Projector": ["projector"],
    "Network/Wi-Fi": ["wi-fi", "wifi", "access point", "lan port"],
    "TV/Display": ["smart tv", "tv screen", "monitor"],
    "Chair": ["chair", "armchair"],
    "Table/Desk": ["table", "desk", "podium"],
    "Whiteboard": ["whiteboard", "chalkboard"],
    "Window": ["window", "jalousie"],
    "Door": ["door", "doorknob"],
    "Ceiling/Roof": ["ceiling board", "ceiling panel", "roof"],
    "Floor": ["floor tile", "floor is", "tiled floor"],
    "Faucet/Sink": ["faucet", "sink"],
    "Drain/Pipe": ["drain", "pipe"],
}

def extract_items(text):
    t = text.lower()
    return [tag for tag, kws in KEYWORDS.items()
            if any(re.search(r"\b" + re.escape(k), t) for k in kws)]

@st.cache_data
def load_data():
    df = pd.read_csv("data/classroom_issues.csv", parse_dates=["timestamp"])
    df["items"] = df["issue_description"].apply(extract_items)
    return df

df_all = load_data()

st.title("Classroom Facility Issue Dashboard - ADSSU")
st.caption("Sample dataset for prototype purposes. Replace data/classroom_issues.csv with real data.")

# ---------- Filters ----------
st.sidebar.header("Filters")
buildings = st.sidebar.multiselect("Building", sorted(df_all["building_name"].unique()), sorted(df_all["building_name"].unique()))
cats = st.sidebar.multiselect("Category", sorted(df_all["category"].unique()), sorted(df_all["category"].unique()))
df = df_all[df_all["building_name"].isin(buildings) & df_all["category"].isin(cats)]

tab1, tab2, tab3, tab4 = st.tabs(["1. Data Exploration", "2. Text Classification", "3. Association Rules", "4. Decision Support"])

# ---------- 1. Exploration ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", len(df))
    c2.metric("Open (Pending / In Progress)", int((df["resolution_status"] != "Resolved").sum()))
    c3.metric("Critical reports", int((df["urgency_level"] == "Critical").sum()))
    c4.metric("Avg days to resolve", round(df["days_to_resolve"].mean(), 1) if len(df) else 0)

    order = ["Low", "Medium", "High", "Critical"]
    st.plotly_chart(px.histogram(df, x="category", color="urgency_level", category_orders={"urgency_level": order}, title="Reports by category and urgency"), use_container_width=True)
    weekly = df.groupby(df["timestamp"].dt.to_period("W").astype(str)).size().reset_index(name="reports")
    st.plotly_chart(px.line(weekly, x="timestamp", y="reports", markers=True, title="Weekly reports"), use_container_width=True)
    st.plotly_chart(px.box(df.dropna(subset=["days_to_resolve"]), x="category", y="days_to_resolve", title="Days to resolve by category"), use_container_width=True)
    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("missing").to_frame())
    st.caption("days_to_resolve is blank only for reports that are not yet Resolved.")

# ---------- 2. Text classification ----------
with tab2:
    st.write("Predicts the category of a report from its description (TF-IDF + Logistic Regression).")
    X_train, X_test, y_train, y_test = train_test_split(df_all["issue_description"], df_all["category"], test_size=0.2, random_state=42, stratify=df_all["category"])
    vec = TfidfVectorizer(ngram_range=(1, 2))
    model = LogisticRegression(max_iter=2000).fit(vec.fit_transform(X_train), y_train)
    pred = model.predict(vec.transform(X_test))
    st.metric("Accuracy (20% test set)", round(accuracy_score(y_test, pred), 3))
    st.text(classification_report(y_test, pred))
    text = st.text_area("Try a new report", "Aircon is leaking water near the wall outlet")
    if text:
        probs = pd.Series(model.predict_proba(vec.transform([text]))[0], index=model.classes_).sort_values(ascending=False)
        st.success(f"Predicted category: {probs.index[0]}")
        st.bar_chart(probs)

# ---------- 3. Association rules ----------
with tab3:
    st.write("Finds facility items that are often reported together in the same report.")
    ms = st.slider("Minimum support", 0.01, 0.20, 0.02, 0.005)
    ml = st.slider("Minimum lift", 1.0, 3.0, 1.2, 0.1)
    basket = pd.DataFrame({tag: df["items"].apply(lambda x, t=tag: t in x) for tag in KEYWORDS})
    freq = apriori(basket, min_support=ms, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=ml) if len(freq) > 1 else pd.DataFrame()
    if rules.empty:
        st.info("No rules found. Lower the minimum support or lift.")
    else:
        rules["if reported"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["also reported"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
        st.dataframe(rules[["if reported", "also reported", "support", "confidence", "lift"]].sort_values("lift", ascending=False).round(3))
        st.caption("Lift above 1 means the items appear together more often than by chance.")

# ---------- 4. Decision support ----------
with tab4:
    st.write("Rooms ranked for maintenance priority.")
    weight = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    d = df.copy()
    d["urgency_points"] = d["urgency_level"].map(weight)
    d["is_open"] = (d["resolution_status"] != "Resolved").astype(int)
    pri = d.groupby(["building_name", "room_number"]).agg(
        reports=("report_id", "count"),
        urgency_points=("urgency_points", "sum"),
        open_reports=("is_open", "sum"),
        avg_days=("days_to_resolve", "mean"),
    ).reset_index()
    pri["avg_days"] = pri["avg_days"].fillna(0).round(1)
    pri["priority_score"] = (pri["urgency_points"] + pri["open_reports"] * 3 + pri["avg_days"] / 2).round(1)
    pri = pri.sort_values("priority_score", ascending=False)
    pri["recommended_action"] = pd.cut(pri["priority_score"].rank(pct=True), bins=[0, .6, .85, 1.0], labels=["Routine monitoring", "Schedule inspection", "Immediate maintenance"])
    st.dataframe(pri, use_container_width=True)
    st.download_button("Download priority report (CSV)", pri.to_csv(index=False), "priority_report.csv", "text/csv")
