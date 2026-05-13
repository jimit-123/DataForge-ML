# =============================================================================
# app.py — Streamlit UI for the ML Data Preprocessing Tool
# =============================================================================
# Run with:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_dataset,
    summarise_dataset,
    df_to_csv_bytes,
    get_numeric_columns,
    get_categorical_columns,
    fmt_number,
)
from preprocessing import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    run_full_pipeline,
    split_dataset,
)

# =============================================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="ML Data Preprocessor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL STYLE OVERRIDES
# =============================================================================
st.markdown(
    """
    <style>
        /* Slightly wider content area */
        .block-container { padding-top: 1.5rem; }

        /* Metric card tweaks */
        [data-testid="metric-container"] {
            background: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 10px;
            padding: 0.6rem 1rem;
        }

        /* Section header style */
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #cba6f7;
            border-left: 4px solid #cba6f7;
            padding-left: 0.6rem;
            margin-bottom: 0.8rem;
        }

        /* Small badge */
        .badge {
            display: inline-block;
            background: #313244;
            color: #cdd6f4;
            border-radius: 4px;
            padding: 1px 7px;
            font-size: 0.78rem;
            margin: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SESSION-STATE INITIALISATION
# =============================================================================
# Streamlit re-runs the entire script on every interaction.
# We store important objects in st.session_state so they persist.

for key, default in {
    "raw_df":       None,   # Original uploaded DataFrame
    "cleaned_df":   None,   # DataFrame after preprocessing
    "full_report":  None,   # Step-by-step preprocessing report
    "splits":       None,   # Train / Val / Test DataFrames
    "split_report": None,   # Split size info
    "file_name":    "",     # Uploaded file name
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# HELPERS
# =============================================================================

def section(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def metric_row(metrics: dict):
    """Render a row of st.metric cards from a dict {label: value}."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def show_report_expander(title: str, report: dict):
    """Generic expander that pretty-prints a report dict."""
    with st.expander(title, expanded=False):
        st.json(report)


# =============================================================================
# SIDEBAR  (settings panel)
# =============================================================================

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/machine-learning.png",
        width=60,
    )
    st.title("⚙️ Settings")
    st.caption("Configure every step of the preprocessing pipeline.")

    st.divider()

    # ── Missing value strategy ───────────────────────────────────────────
    st.subheader("🩹 Missing Values")
    missing_strategy = st.radio(
        "Fill strategy for numeric columns",
        options=["auto", "mean", "median"],
        index=0,
        help="'auto' uses the median, which is more robust to skewed data.",
    )

    st.divider()

    # ── Outlier settings ─────────────────────────────────────────────────
    st.subheader("📊 Outlier Handling")
    outlier_method = st.radio(
        "Detection method",
        options=["iqr", "zscore"],
        index=0,
        help="IQR is robust to skewed distributions. Z-score assumes normality.",
    )
    outlier_action = st.radio(
        "What to do with outliers",
        options=["cap", "remove"],
        index=0,
        help="'cap' (Winsorisation) preserves row count. 'remove' drops rows.",
    )

    st.divider()

    # ── Encoding ─────────────────────────────────────────────────────────
    st.subheader("🔢 Categorical Encoding")
    encoding_method = st.radio(
        "Encoding strategy",
        options=["label", "onehot"],
        index=0,
        help=(
            "Label: each category → integer (compact, good for trees). "
            "One-Hot: binary column per category (better for linear models)."
        ),
    )

    st.divider()

    # ── Scaling ──────────────────────────────────────────────────────────
    st.subheader("📐 Feature Scaling")
    scaling_method = st.radio(
        "Scaling strategy",
        options=["standard", "minmax"],
        index=0,
        help=(
            "StandardScaler → zero mean, unit variance. "
            "MinMaxScaler   → scales to [0, 1]."
        ),
    )

    st.divider()

    # ── Overfitting prevention ────────────────────────────────────────────
    st.subheader("🛡️ Overfitting Prevention")

    remove_correlated = st.checkbox("Remove highly correlated features", value=True)
    corr_threshold = st.slider(
        "Correlation threshold",
        min_value=0.70, max_value=1.00, value=0.90, step=0.01,
        disabled=not remove_correlated,
    )

    remove_low_var = st.checkbox("Remove low-variance features", value=False)
    var_threshold = st.slider(
        "Variance threshold",
        min_value=0.0, max_value=0.10, value=0.01, step=0.005,
        disabled=not remove_low_var,
        format="%.3f",
    )

    apply_pca = st.checkbox("Apply PCA dimensionality reduction", value=False)
    pca_variance = st.slider(
        "Variance to retain (%)",
        min_value=80, max_value=99, value=95,
        disabled=not apply_pca,
        help="PCA will keep enough components to explain this % of variance.",
    )

    st.divider()

    # ── Train / Val / Test split ──────────────────────────────────────────
    st.subheader("✂️ Dataset Split")
    test_size = st.slider("Test set size",  min_value=5,  max_value=30, value=15, step=5)
    val_size  = st.slider("Val set size",   min_value=5,  max_value=30, value=15, step=5)
    st.caption(f"Train will be **{100 - test_size - val_size} %** of the data.")

    # Pack all settings into one dict for easy passing to the pipeline
    PIPELINE_CONFIG = {
        "missing_strategy":  missing_strategy,
        "outlier_method":    outlier_method,
        "outlier_action":    outlier_action,
        "encoding_method":   encoding_method,
        "scaling_method":    scaling_method,
        "remove_correlated": remove_correlated,
        "corr_threshold":    corr_threshold,
        "remove_low_var":    remove_low_var,
        "var_threshold":     var_threshold,
        "apply_pca":         apply_pca,
        "pca_variance":      pca_variance / 100,
        "test_size":         test_size  / 100,
        "val_size":          val_size   / 100,
        "target_col":        None,      # filled in the main area below
    }


# =============================================================================
# MAIN AREA  — Tabs
# =============================================================================

st.title("🤖 ML Data Preprocessing Tool")
st.caption(
    "Upload a raw dataset → let the pipeline clean, encode, scale, and split it "
    "automatically → download model-ready data."
)

tab_upload, tab_clean, tab_outlier, tab_pipeline, tab_dashboard = st.tabs([
    "📁 Upload",
    "🧹 Cleaning Report",
    "📊 Outlier Report",
    "🚀 Pipeline & Split",
    "📈 Dashboard",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    section("Upload Your Dataset")

    uploaded = st.file_uploader(
        "Drop a CSV, Excel (.xlsx), or JSON file here (max 20 MB)",
        type=["csv", "xlsx", "xls", "json"],
        help="The file will be validated for size and format before loading.",
    )

    if uploaded:
        with st.spinner("Loading and validating file …"):
            df, error = load_dataset(uploaded)

        if error:
            st.error(f"❌ {error}")
        else:
            st.session_state.raw_df    = df
            st.session_state.file_name = uploaded.name
            # Clear previous pipeline results when a new file is uploaded
            st.session_state.cleaned_df  = None
            st.session_state.full_report = None
            st.session_state.splits      = None
            st.session_state.split_report = None
            st.success(f"✅ **{uploaded.name}** loaded successfully!")

    # ── Dataset overview ─────────────────────────────────────────────────
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        summary = summarise_dataset(df)

        st.divider()
        section("Dataset Overview")

        metric_row({
            "Rows":              fmt_number(summary["rows"]),
            "Columns":           fmt_number(summary["columns"]),
            "Missing Values":    f"{summary['missing_cells']} ({summary['missing_pct']} %)",
            "Duplicate Rows":    fmt_number(summary["duplicate_rows"]),
            "Numeric Columns":   fmt_number(summary["numeric_cols"]),
            "Categorical Cols":  fmt_number(summary["categorical_cols"]),
        })

        st.divider()
        section("Data Preview (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True, height=350)

        st.divider()
        section("Column Info")
        # Build a neat info table
        info_rows = []
        for col in df.columns:
            info_rows.append({
                "Column":     col,
                "Dtype":      str(df[col].dtype),
                "Non-Null":   int(df[col].notna().sum()),
                "Null %":     f"{df[col].isna().mean() * 100:.1f} %",
                "Unique":     int(df[col].nunique()),
                "Sample":     str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(info_rows), use_container_width=True, height=300)

        # ── Missing-value heatmap ──────────────────────────────────────
        if summary["missing_cells"] > 0:
            st.divider()
            section("Missing Values Heatmap")
            miss = df.isnull().mean().reset_index()
            miss.columns = ["Column", "Missing Fraction"]
            miss = miss[miss["Missing Fraction"] > 0].sort_values("Missing Fraction", ascending=True)

            fig = px.bar(
                miss, x="Missing Fraction", y="Column", orientation="h",
                color="Missing Fraction",
                color_continuous_scale="Reds",
                title="Fraction of Missing Values per Column",
            )
            fig.update_layout(height=max(300, 30 * len(miss)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Upload a file above to get started.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CLEANING REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_clean:
    section("Run the Cleaning Pipeline")

    if st.session_state.raw_df is None:
        st.info("📁 Please upload a dataset first (Upload tab).")
    else:
        df_raw = st.session_state.raw_df

        # ── Target column selector ────────────────────────────────────
        st.markdown("**Select the target column** (what you want to predict):")
        all_cols = ["— None —"] + df_raw.columns.tolist()
        target_choice = st.selectbox(
            "Target column",
            options=all_cols,
            index=0,
            help="The target column will be excluded from encoding and scaling.",
        )
        target_col = None if target_choice == "— None —" else target_choice
        PIPELINE_CONFIG["target_col"] = target_col

        if st.button("▶️ Run Full Preprocessing Pipeline", type="primary"):
            with st.spinner("Preprocessing … this may take a moment for large datasets."):
                try:
                    cleaned_df, full_report = run_full_pipeline(df_raw.copy(), PIPELINE_CONFIG)
                    st.session_state.cleaned_df  = cleaned_df
                    st.session_state.full_report  = full_report
                    st.session_state.splits       = None  # Reset splits
                    st.session_state.split_report = None
                    st.success("✅ Pipeline complete! Check the other tabs for details.")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    raise

        # ── Show cleaning report ──────────────────────────────────────
        if st.session_state.full_report:
            rpt = st.session_state.full_report
            cleaned = st.session_state.cleaned_df

            st.divider()
            section("Cleaning Summary")

            raw_shape = df_raw.shape
            clean_shape = cleaned.shape
            dup_removed = rpt["duplicates"]["duplicates_removed"]
            cols_dropped_null = rpt["useless_cols"]["columns_dropped"]
            filled_cols = rpt["missing_values"]["filled_columns"]
            retyped = rpt["type_fixes"]["columns_retyped"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows (before → after)",
                        f"{raw_shape[0]:,} → {clean_shape[0]:,}",
                        delta=f"-{raw_shape[0] - clean_shape[0]:,}")
            col2.metric("Columns (before → after)",
                        f"{raw_shape[1]:,} → {clean_shape[1]:,}",
                        delta=f"-{raw_shape[1] - clean_shape[1]:,}")
            col3.metric("Duplicates removed", fmt_number(dup_removed))
            col4.metric("Columns imputed", fmt_number(len(filled_cols)))

            st.divider()

            # ── Step-by-step breakdown ─────────────────────────────
            section("Step-by-Step Breakdown")

            with st.expander("1️⃣ Duplicate Removal", expanded=True):
                d = rpt["duplicates"]
                st.write(f"Found and removed **{d['duplicates_removed']:,}** duplicate rows.")

            with st.expander("2️⃣ Data Type Fixes", expanded=True):
                retyped = rpt["type_fixes"]["columns_retyped"]
                if retyped:
                    for r in retyped:
                        st.write(f"• {r}")
                else:
                    st.write("No type corrections needed.")

            with st.expander("3️⃣ Useless Column Removal", expanded=True):
                dropped = rpt["useless_cols"]["columns_dropped"]
                if dropped:
                    for d in dropped:
                        st.write(f"• {d}")
                else:
                    st.write("All columns retained.")

            with st.expander("4️⃣ Text Normalisation", expanded=True):
                normed = rpt["text_normalisation"]["normalised_columns"]
                if normed:
                    st.write(f"Normalised (strip / lowercase): **{', '.join(normed)}**")
                else:
                    st.write("No text normalisation needed.")

            with st.expander("5️⃣ Missing Value Imputation", expanded=True):
                if filled_cols:
                    rows = []
                    for col, info in filled_cols.items():
                        rows.append({
                            "Column":      col,
                            "Missing":     info["missing"],
                            "Method":      info["method"],
                            "Fill Value":  info["fill_value"],
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("No missing values found.")

            with st.expander("6️⃣ Encoding", expanded=True):
                enc = rpt["encoding"]["encoded_columns"]
                if enc:
                    rows = [{"Column": k, "Encoding": v} for k, v in enc.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("No categorical columns to encode.")

            with st.expander("7️⃣ Feature Correlation (Overfitting Prevention)", expanded=True):
                if "correlation" in rpt:
                    c = rpt["correlation"]
                    st.write(f"Threshold: **{c['threshold']}** | "
                             f"Pairs found: **{c['correlated_pairs_found']}** | "
                             f"Columns dropped: **{len(c['columns_dropped'])}**")
                    if c["pairs"]:
                        pair_df = pd.DataFrame(c["pairs"], columns=["Column A", "Column B", "Correlation"])
                        st.dataframe(pair_df, use_container_width=True, hide_index=True)
                else:
                    st.write("Correlation removal not enabled.")

            with st.expander("8️⃣ Scaling", expanded=True):
                s = rpt["scaling"]
                st.write(f"Method: **{s['method']}** | "
                         f"Scaled **{len(s['scaled_columns'])}** columns.")
                if s["scaled_columns"]:
                    st.write("Columns: " + ", ".join(f"`{c}`" for c in s["scaled_columns"]))

            if "pca" in rpt:
                with st.expander("9️⃣ PCA", expanded=True):
                    p = rpt["pca"]
                    if p["applied"]:
                        st.write(
                            f"Reduced **{p['original_features']}** features → "
                            f"**{p['components_kept']}** components "
                            f"(explains **{p['variance_explained']} %** of variance)."
                        )
                    else:
                        st.write(p.get("reason", "PCA not applied."))

            st.divider()
            section("Cleaned Dataset Preview")
            st.dataframe(cleaned.head(100), use_container_width=True, height=300)

            # ── Download cleaned dataset ───────────────────────────
            st.download_button(
                label="⬇️ Download Cleaned Dataset (CSV)",
                data=df_to_csv_bytes(cleaned),
                file_name=f"cleaned_{st.session_state.file_name.rsplit('.', 1)[0]}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — OUTLIER REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_outlier:
    section("Outlier Analysis")

    if st.session_state.raw_df is None:
        st.info("📁 Please upload a dataset first (Upload tab).")
    else:
        df_raw = st.session_state.raw_df
        num_cols = get_numeric_columns(df_raw)

        if not num_cols:
            st.warning("No numeric columns found for outlier analysis.")
        else:
            col_l, col_r = st.columns(2)

            # ── IQR results ───────────────────────────────────────────
            with col_l:
                section("IQR Method")
                iqr_counts = detect_outliers_iqr(df_raw)
                if iqr_counts:
                    iqr_df = pd.DataFrame(
                        [(k, v) for k, v in iqr_counts.items()],
                        columns=["Column", "Outliers (IQR)"]
                    ).sort_values("Outliers (IQR)", ascending=False)
                    st.dataframe(iqr_df, use_container_width=True, hide_index=True)
                    st.metric("Total IQR outliers", fmt_number(sum(iqr_counts.values())))
                else:
                    st.success("No IQR outliers detected.")

            # ── Z-Score results ───────────────────────────────────────
            with col_r:
                section("Z-Score Method (|z| > 3)")
                z_counts = detect_outliers_zscore(df_raw)
                if z_counts:
                    z_df = pd.DataFrame(
                        [(k, v) for k, v in z_counts.items()],
                        columns=["Column", "Outliers (Z-score)"]
                    ).sort_values("Outliers (Z-score)", ascending=False)
                    st.dataframe(z_df, use_container_width=True, hide_index=True)
                    st.metric("Total Z-score outliers", fmt_number(sum(z_counts.values())))
                else:
                    st.success("No Z-score outliers detected.")

            st.divider()

            # ── Box plots ─────────────────────────────────────────────
            section("Box Plots — Numeric Columns")
            # Allow the user to pick which columns to visualise (avoid clutter)
            display_cols = st.multiselect(
                "Columns to display",
                options=num_cols,
                default=num_cols[:min(6, len(num_cols))],
            )

            if display_cols:
                n_cols = min(3, len(display_cols))
                n_rows = (len(display_cols) + n_cols - 1) // n_cols

                fig = make_subplots(rows=n_rows, cols=n_cols,
                                    subplot_titles=display_cols)

                for i, col in enumerate(display_cols):
                    row = i // n_cols + 1
                    col_idx = i % n_cols + 1
                    fig.add_trace(
                        go.Box(y=df_raw[col].dropna(), name=col, showlegend=False),
                        row=row, col=col_idx,
                    )

                fig.update_layout(
                    height=300 * n_rows,
                    title_text="Box Plots (raw data)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Before / after comparison ─────────────────────────────
            if st.session_state.cleaned_df is not None:
                st.divider()
                section("Before vs After Outlier Handling")
                cleaned = st.session_state.cleaned_df
                compare_cols = [c for c in display_cols if c in cleaned.columns]

                if compare_cols:
                    sel_col = st.selectbox("Select column to compare", compare_cols)
                    fig2 = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=["Before preprocessing", "After preprocessing"],
                    )
                    fig2.add_trace(
                        go.Histogram(x=df_raw[sel_col].dropna(), name="Before",
                                     marker_color="#f38ba8"), row=1, col=1)
                    if sel_col in cleaned.columns:
                        fig2.add_trace(
                            go.Histogram(x=cleaned[sel_col].dropna(), name="After",
                                         marker_color="#a6e3a1"), row=1, col=2)
                    fig2.update_layout(height=380, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PIPELINE & SPLIT
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipeline:
    section("Train / Validation / Test Split")

    if st.session_state.cleaned_df is None:
        st.info("🧹 Run the preprocessing pipeline first (Cleaning Report tab).")
    else:
        cleaned = st.session_state.cleaned_df
        target_col = PIPELINE_CONFIG.get("target_col")

        if not target_col:
            st.warning(
                "⚠️ No target column was selected. "
                "Go back to the Cleaning Report tab, pick a target, and re-run the pipeline."
            )
        elif target_col not in cleaned.columns:
            st.warning(
                f"⚠️ Target column **{target_col}** is not present after preprocessing "
                "(it may have been encoded or dropped). Please re-run the pipeline."
            )
        else:
            st.info(
                f"Target column: **{target_col}** | "
                f"Features: **{cleaned.shape[1] - 1}** | "
                f"Rows: **{cleaned.shape[0]:,}**"
            )

            if st.button("✂️ Split Dataset", type="primary"):
                with st.spinner("Splitting …"):
                    try:
                        splits, split_report = split_dataset(
                            cleaned,
                            target_col=target_col,
                            test_size=PIPELINE_CONFIG["test_size"],
                            val_size=PIPELINE_CONFIG["val_size"],
                        )
                        st.session_state.splits = splits
                        st.session_state.split_report = split_report
                        st.success("✅ Dataset split successfully!")
                    except Exception as exc:
                        st.error(f"Split error: {exc}")

        if st.session_state.split_report:
            rpt = st.session_state.split_report
            splits = st.session_state.splits

            st.divider()
            section("Split Summary")

            metric_row({
                "Total Rows":      fmt_number(rpt["total_rows"]),
                "Features":        fmt_number(rpt["n_features"]),
                f"Train ({rpt['train_pct']} %)":  fmt_number(rpt["train_rows"]),
                f"Val   ({rpt['val_pct']} %)":    fmt_number(rpt["val_rows"]),
                f"Test  ({rpt['test_pct']} %)":   fmt_number(rpt["test_rows"]),
            })

            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=["Train", "Validation", "Test"],
                values=[rpt["train_rows"], rpt["val_rows"], rpt["test_rows"]],
                hole=0.4,
                marker_colors=["#89b4fa", "#a6e3a1", "#f38ba8"],
            )])
            fig.update_layout(title="Dataset Split Distribution", height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            section("Set Previews")

            split_tabs = st.tabs(["Training Set", "Validation Set", "Test Set"])

            set_labels = [
                ("Training Set",   splits["X_train"], splits["y_train"]),
                ("Validation Set", splits["X_val"],   splits["y_val"]),
                ("Test Set",       splits["X_test"],  splits["y_test"]),
            ]

            for stab, (label, X, y) in zip(split_tabs, set_labels):
                with stab:
                    preview = X.copy()
                    preview[target_col] = y.values
                    st.dataframe(preview.head(50), use_container_width=True, height=250)

                    csv_bytes = df_to_csv_bytes(preview)
                    fname = label.lower().replace(" ", "_") + ".csv"
                    st.download_button(
                        f"⬇️ Download {label}",
                        data=csv_bytes,
                        file_name=fname,
                        mime="text/csv",
                        key=f"dl_{label}",
                    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    section("Performance Dashboard")

    if st.session_state.raw_df is None:
        st.info("📁 Upload a dataset to see the dashboard.")
    else:
        df_raw = st.session_state.raw_df
        cleaned = st.session_state.cleaned_df

        # ── Before / After summary ────────────────────────────────────
        section("Before vs After Preprocessing")

        raw_sum = summarise_dataset(df_raw)
        cl_sum  = summarise_dataset(cleaned) if cleaned is not None else None

        b_col, a_col = st.columns(2)
        with b_col:
            st.markdown("**🔴 Raw Dataset**")
            st.metric("Rows",            fmt_number(raw_sum["rows"]))
            st.metric("Columns",         fmt_number(raw_sum["columns"]))
            st.metric("Missing cells",   fmt_number(raw_sum["missing_cells"]))
            st.metric("Duplicate rows",  fmt_number(raw_sum["duplicate_rows"]))
            st.metric("Memory (KB)",     fmt_number(raw_sum["memory_kb"]))

        with a_col:
            st.markdown("**🟢 Cleaned Dataset**")
            if cl_sum:
                st.metric("Rows",           fmt_number(cl_sum["rows"]))
                st.metric("Columns",        fmt_number(cl_sum["columns"]))
                st.metric("Missing cells",  fmt_number(cl_sum["missing_cells"]))
                st.metric("Duplicate rows", fmt_number(cl_sum["duplicate_rows"]))
                st.metric("Memory (KB)",    fmt_number(cl_sum["memory_kb"]))
            else:
                st.info("Run the pipeline to see cleaned stats.")

        st.divider()

        # ── Numeric distributions ─────────────────────────────────────
        num_cols = get_numeric_columns(df_raw)
        if num_cols:
            section("Numeric Feature Distributions (raw)")
            cols_to_show = num_cols[:9]  # Show at most 9 to avoid clutter
            n_c = min(3, len(cols_to_show))
            n_r = (len(cols_to_show) + n_c - 1) // n_c
            fig = make_subplots(rows=n_r, cols=n_c, subplot_titles=cols_to_show)
            for idx, col in enumerate(cols_to_show):
                r = idx // n_c + 1
                c = idx % n_c + 1
                fig.add_trace(
                    go.Histogram(x=df_raw[col].dropna(), name=col, showlegend=False),
                    row=r, col=c,
                )
            fig.update_layout(height=300 * n_r, title_text="Histograms (raw data)")
            st.plotly_chart(fig, use_container_width=True)

        # ── Correlation heatmap ───────────────────────────────────────
        if len(num_cols) > 1:
            st.divider()
            section("Correlation Heatmap (raw numeric features)")
            corr_matrix = df_raw[num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Pearson Correlation Matrix",
                height=max(400, 50 * len(num_cols)),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Categorical value counts ───────────────────────────────────
        cat_cols = get_categorical_columns(df_raw)
        if cat_cols:
            st.divider()
            section("Categorical Column Value Counts")
            sel_cat = st.selectbox("Select a categorical column", cat_cols)
            vc = df_raw[sel_cat].value_counts().head(20).reset_index()
            vc.columns = [sel_cat, "Count"]
            fig_bar = px.bar(
                vc, x=sel_cat, y="Count",
                color="Count", color_continuous_scale="Purples",
                title=f"Top 20 values in '{sel_cat}'",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Full pipeline report (raw JSON) ───────────────────────────
        if st.session_state.full_report:
            st.divider()
            section("Full Pipeline Report (JSON)")
            with st.expander("View complete report", expanded=False):
                st.json(st.session_state.full_report)
