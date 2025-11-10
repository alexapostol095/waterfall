import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ======================================================
# 1️⃣ Page Setup
# ======================================================
st.set_page_config(page_title="Q3 Waterfall Comparison", layout="wide")
st.title("📊 Q3 2024 → Q3 2025 Comparison Dashboard")

st.write(
    "Upload your **comp** dataset (CSV or Excel). "
    "You can define two groups (e.g., sensitivity clusters or product classes) and compare their price & volume effects side by side."
)

# ======================================================
# 2️⃣ Upload
# ======================================================
uploaded_file = st.file_uploader("📂 Upload comp dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("✅ File loaded successfully!")

    # ======================================================
    # 3️⃣ Filter Definitions (NO auto-select)
    # ======================================================
    st.subheader("🎛️ Define Comparison Groups")

    # Identify categorical columns
    filter_cols = [c for c in df.columns if c.startswith("ProdClass") or "Sensitivity" in c]
    if not filter_cols:
        st.warning("No suitable categorical columns found for filtering (e.g. ProdClass or SensitivityCluster).")

    colA, colB = st.columns(2)
    group_filters = {}

    for group_name, col_target in zip(["Group A", "Group B"], [colA, colB]):
        with col_target:
            st.markdown(f"### {group_name}")
            group_filters[group_name] = {}
            for col in filter_cols:
                unique_vals = sorted(df[col].dropna().unique())
                # 🔹 No defaults selected
                selected = st.multiselect(
                    f"{col} ({group_name}):",
                    unique_vals,
                    default=[]
                )
                group_filters[group_name][col] = selected

    # ======================================================
    # 4️⃣ Metric Selection
    # ======================================================
    st.divider()
    metric_type = st.radio("Select metric to visualise:", ["Revenue", "Margin"], horizontal=True)

    # ======================================================
    # 5️⃣ Filter Application
    # ======================================================
    def apply_filters(df, filters):
        df_filt = df.copy()
        for col, vals in filters.items():
            if vals:
                df_filt = df_filt[df_filt[col].isin(vals)]
        return df_filt

    groupA_df = apply_filters(df, group_filters["Group A"])
    groupB_df = apply_filters(df, group_filters["Group B"])

    st.markdown(f"**Group A Rows:** {len(groupA_df)}  |  **Group B Rows:** {len(groupB_df)}")

    # ======================================================
    # 6️⃣ Aggregation Function
    # ======================================================
    def get_metrics(df, metric_type):
        if len(df) == 0:
            return 0, 0, 0, 0

        if metric_type == "Revenue":
            start = df["Rev_Q3_2024_base"].sum()
            end = df["Rev_Q3_2025_actual"].sum()
            vol = df["Vol_Contribution"].sum()
            price = df["Price_Contribution"].sum()
        else:
            start = df["Mar_Q3_2024_base"].sum()
            end = df["Mar_Q3_2025_actual"].sum()
            vol = df["Vol_Contribution_Mar"].sum()
            price = df["Price_Contribution_Mar"].sum()
        return start, vol, price, end

    # ======================================================
    # 7️⃣ Matplotlib Waterfall Plot
    # ======================================================
    def plot_waterfall(start, end, vol, price, metric_name, title_suffix=""):
        steps = [
            (f"Start {metric_name} Q3 2024", start),
            ("Volume Effect", vol),
            ("Price Effect", price),
            (f"End {metric_name} Q3 2025", end)
        ]

        x = np.arange(len(steps))
        labels = [s[0] for s in steps]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#F9F6F0")
        ax.set_facecolor("#F9F6F0")

        # --- Start bar ---
        ax.bar(0, start, color="#0B0B45", width=0.6)

        # --- Effects ---
        running_total = start
        for i, (label, effect) in enumerate(steps[1:-1], start=1):
            color = "#3C37FF" if effect >= 0 else "#FF6F61"
            ax.bar(i, effect, bottom=running_total if effect < 0 else running_total,
                   color=color, width=0.6)
            running_total += effect

            pct = (effect / start) * 100 if start != 0 else 0
            ax.text(i,
                    running_total + (start * 0.03 if effect >= 0 else -start * 0.05),
                    f"{effect:,.0f} | {pct:+.1f}%",
                    ha="center",
                    va="bottom" if effect >= 0 else "top",
                    fontsize=10, fontweight="medium")

        # --- Final bar ---
        ax.bar(len(steps)-1, end, color="#0B0B45", width=0.6)

        # --- Labels for start and end ---
        ax.text(0, start + start * 0.03,
                f"{start:,.0f} | Baseline",
                ha="center", va="bottom", fontsize=10, fontweight="medium")

        total_pct = (end - start) / start * 100 if start != 0 else 0
        ax.text(len(steps)-1, end + start * 0.03,
                f"{end:,.0f} | {total_pct:+.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="medium")

        # --- Styling ---
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel(f"{metric_name} (€)")
        ax.set_title(f"{metric_name} Bridge (Q3 2024 → Q3 2025) {title_suffix}", fontweight="bold")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        plt.tight_layout()
        return fig

    # ======================================================
    # 8️⃣ Compute Metrics and Display Charts
    # ======================================================
    startA, volA, priceA, endA = get_metrics(groupA_df, metric_type)
    startB, volB, priceB, endB = get_metrics(groupB_df, metric_type)

    # ======================================================
    # 🧭 1️⃣ SUMMARY KPIs SECTION
    # ======================================================
    st.subheader("📈 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Start {metric_type} (Group A)", f"{startA:,.0f} €")
    col2.metric(f"End {metric_type} (Group A)", f"{endA:,.0f} €", f"{(endA-startA)/startA*100 if startA else 0:+.1f}%")
    col3.metric(f"Start {metric_type} (Group B)", f"{startB:,.0f} €")
    col4.metric(f"End {metric_type} (Group B)", f"{endB:,.0f} €", f"{(endB-startB)/startB*100 if startB else 0:+.1f}%")

    st.divider()

    # ======================================================
    # 9️⃣ Display Charts
    # ======================================================
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟦 Group A")
        figA = plot_waterfall(startA, endA, volA, priceA, metric_type, title_suffix="– Group A")
        st.pyplot(figA)

    with col2:
        st.markdown("### 🟥 Group B")
        figB = plot_waterfall(startB, endB, volB, priceB, metric_type, title_suffix="– Group B")
        st.pyplot(figB)

    # ======================================================
    # 🔟 5️⃣ AUTOMATED INSIGHT SECTION
    # ======================================================
    deltaA = (endA - startA) / startA * 100 if startA else 0
    deltaB = (endB - startB) / startB * 100 if startB else 0
    delta_diff = deltaB - deltaA

    st.subheader("🧠 Automated Insights")
    st.markdown(f"""
    - **Group A** {metric_type.lower()} changed by **{deltaA:+.1f}%**
    - **Group B** {metric_type.lower()} changed by **{deltaB:+.1f}%**
    - **Difference (B vs A):** {delta_diff:+.1f}%

    Interpretation:
    > {("Group B outperformed Group A" if delta_diff > 0 else "Group A outperformed Group B" if delta_diff < 0 else "Both groups performed similarly")}.
    {metric_type} differences are driven primarily by the relative strength of volume and price effects shown in the waterfalls above.
    """)

    # ======================================================
    # 🔁 Downloads
    # ======================================================
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csvA = groupA_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Group A (CSV)", csvA, "groupA_filtered.csv", "text/csv")

    with col_dl2:
        csvB = groupB_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Group B (CSV)", csvB, "groupB_filtered.csv", "text/csv")

else:
    st.info("👆 Upload your `comp` dataset to begin.")
