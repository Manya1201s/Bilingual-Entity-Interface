"""
Shared CSS for every page of the Streamlit app (app.py and pages/).
Injected with st.markdown(APP_CSS, unsafe_allow_html=True).
"""

APP_CSS = """
    <style>
    .stApp {
        background: #0b1220;
        color: #e8eef8;
    }

    [data-testid="stSidebar"] {
        background: #101b2d;
    }

    h1, h2, h3 {
        color: #f4f8ff;
    }

    /* ---- Layout and typography ---- */

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em;
        margin-top: 0.4rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #8ea3bd;
    }

    hr {
        border-color: #1d2e45 !important;
        margin: 1.6rem 0 !important;
    }

    /* ---- Hero ---- */

    .hero {
        padding: 26px 30px;
        border: 1px solid #263b55;
        border-radius: 16px;
        background: linear-gradient(120deg, #14243a, #102d37);
        margin-bottom: 24px;
    }

    .tag {
        color: #5eead4;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin: 8px 0;
    }

    .hero-sub {
        color: #a9bbd1;
        max-width: 720px;
        line-height: 1.55;
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }

    .chip {
        border: 1px solid #2c4664;
        background: rgba(11, 18, 32, 0.45);
        color: #cfe0f2;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 12.5px;
        font-weight: 600;
    }

    /* ---- Buttons ---- */

    [data-testid="stBaseButton-primary"] {
        background: #14b8a6;
        color: #071820;
        border: 1px solid #14b8a6;
        border-radius: 9px;
        font-weight: 700;
        transition: background 0.15s ease, transform 0.05s ease,
                    box-shadow 0.15s ease;
    }

    [data-testid="stBaseButton-primary"]:hover {
        background: #5eead4;
        border-color: #5eead4;
        color: #071820;
    }

    [data-testid="stBaseButton-primary"]:active {
        transform: translateY(1px);
    }

    [data-testid="stBaseButton-secondary"] {
        background: transparent;
        color: #5eead4;
        border: 1px solid #2c4664;
        border-radius: 9px;
        font-weight: 600;
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        border-color: #14b8a6;
        background: rgba(20, 184, 166, 0.08);
        color: #5eead4;
    }

    [data-testid^="stBaseButton"]:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.45);
    }

    [data-testid^="stBaseButton"]:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }

    /* ---- Metrics ---- */

    [data-testid="stMetric"] {
        border: 1px solid #1f3149;
        border-radius: 12px;
        background: #0f1a2b;
        padding: 12px 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #a9bbd1;
    }

    [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
    }

    /* ---- Tabs, expanders, inputs, tables ---- */

    [data-baseweb="tab"] {
        font-weight: 600;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: #5eead4;
    }

    [data-testid="stExpander"] details {
        border: 1px solid #1f3149;
        border-radius: 12px;
        background: #0e1828;
    }

    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.35);
    }

    /* Fixed NER model shown as static text (no dropdown). The top margin
       matches a widget label, so it lines up with the selector beside it. */
    .static-field {
        margin-top: 1.9rem;
        padding: 0.55rem 0.8rem;
        border: 1px solid #1f3149;
        border-radius: 0.5rem;
        background: #0f1a2b;
        color: #cfe0f2;
        font-size: 0.95rem;
        cursor: default;
        user-select: text;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #1f3149;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ---- Small screens ---- */

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 20px 18px;
        }

        .hero-title {
            font-size: 26px;
        }
    }

    /* ---- Research results page ---- */

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 14px;
        margin: 6px 0 18px;
    }

    .metric-card {
        border: 1px solid #263b55;
        border-radius: 14px;
        background: #101b2d;
        padding: 16px 18px;
    }

    .metric-label {
        color: #a9bbd1;
        font-size: 13px;
        font-weight: 600;
    }

    .metric-value {
        color: #f4f8ff;
        font-size: 30px;
        font-weight: 700;
        margin: 4px 0 6px;
        font-variant-numeric: tabular-nums;
    }

    .metric-value small {
        color: #a9bbd1;
        font-size: 15px;
        font-weight: 600;
    }

    .metric-meta {
        color: #8ea3bd;
        font-size: 12.5px;
        line-height: 1.45;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 8px;
    }

    .badge-done {
        background: #0f3b36;
        color: #5eead4;
    }

    .badge-partial {
        background: #1e2f4f;
        color: #93c5fd;
    }

    .badge-pending {
        background: #3b2f0f;
        color: #fbbf24;
    }

    .status-list {
        margin: 4px 0 0;
        padding-left: 18px;
        color: #d5deea;
        font-size: 14px;
        line-height: 1.6;
    }
    </style>
"""
