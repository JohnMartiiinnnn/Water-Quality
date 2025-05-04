import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import datetime

# ==== CONFIGURATION (Developer Customizable) ====
# Adjust the spacing (in pixels) between the main analysis columns here
DEVELOPER_COLUMN_SPACING_PX = 25

# ==== LOAD AND ENCODE HEADER IMAGE ====
try:
    # Make sure 'taal_lake.png' is in the same directory as your script
    with open("taal_lake.png", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
except FileNotFoundError:
    st.error("Error: 'taal_lake.png' not found. Please ensure the image file is present.")
    img_base64 = None

# ==== LOAD AND ENCODE FONT ====
try:
    # Make sure 'Montserrat-Bold.ttf' is in the same directory
    with open("Montserrat-Bold.ttf", "rb") as f:
        font_base64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    st.warning("Warning: 'Montserrat-Bold.ttf' font not found. Using default sans-serif font.")
    font_base64 = None

# ==== CUSTOM FONT STYLE ====
font_style = ""
if font_base64:
    font_style = f"""
    @font-face {{
        font-family: 'Montserrat';
        src: url("data:font/ttf;base64,{font_base64}") format('truetype');
    }}
    /* Apply font globally if loaded */
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .section-header, .custom-label,
    .streamlit-expanderHeader p, /* Target text within expander header */
    .streamlit-expanderContent div, /* Target text within expander content */
    .stButton button, /* General buttons */
    .stFileUploader label,
    .stMultiSelect label {{
        font-family: 'Montserrat', sans-serif !important;
    }}
    """
else:
    # Fallback font style if Montserrat is not found
    font_style = """
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .section-header, .custom-label,
    .streamlit-expanderHeader p,
    .streamlit-expanderContent div,
    .stButton button,
    .stFileUploader label,
    .stMultiSelect label {{
        font-family: sans-serif; /* Fallback font */
    }}
    """

# ==== SESSION STATE INITIALIZATION ====
# Initialize session state variables if they don't exist
if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False
if "df" not in st.session_state:
    st.session_state["df"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

# ==== SIDEBAR ====
with st.sidebar:
    # Expander for Upload and Overview
    with st.expander("📁 Dataset Upload & Overview", expanded=True): # Start expanded

        # --- File Uploader Logic ---
        # Show uploader ONLY if no file has been successfully uploaded yet
        if not st.session_state["file_uploaded"]:
            st.markdown("<div class='custom-label' style='margin-bottom: 5px;'>Select a CSV file:</div>",
                        unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "",                     # Hide the default label string
                type="csv",
                key="file_uploader",    # Assign a key for internal state management
                label_visibility="collapsed" # Use CSS/Markdown for custom label instead
            )

            # Process the uploaded file
            if uploaded_file is not None:
                try:
                    # Store filename and read CSV into DataFrame
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.session_state["df"] = pd.read_csv(uploaded_file)
                    st.session_state["file_uploaded"] = True # Mark upload as successful
                    st.success("File uploaded!")
                    st.rerun() # Rerun the script immediately to update the UI

                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    # Reset state fully on error to allow re-upload attempt
                    st.session_state["df"] = None
                    st.session_state["file_uploaded"] = False
                    st.session_state["uploaded_filename"] = None
                    # Note: No need to reset 'file_uploader' key here

        # --- Display Loaded File Info and Remove Button ---
        # Show this section ONLY if a file has been uploaded
        else:
            # Display filename if available
            if st.session_state["uploaded_filename"]:
                st.success(f"✅ Loaded: {st.session_state['uploaded_filename']}")
            else:
                st.success("✅ Dataset loaded.") # Fallback message

            # Provide button to remove the dataset and reset state
            if st.button("❌ Remove Dataset"):
                # Reset the relevant state variables
                st.session_state["file_uploaded"] = False
                st.session_state["df"] = None
                st.session_state["uploaded_filename"] = None
                # DO NOT reset the widget key: st.session_state["file_uploader"] = None

                # Rerun the app to reflect the changes (will show uploader again)
                st.rerun()

            # This block executes only if file_uploaded is True AND df is not None
            if st.session_state["file_uploaded"] and st.session_state["df"] is not None:
                st.markdown("---") # Visual separator
                st.markdown("<h3 style='color: #003366; font-size: 17px; margin-bottom: 5px;'>📊 Dataset Overview</h3>", unsafe_allow_html=True)

                # Get the DataFrame from session state
                df_sidebar = st.session_state["df"]

                # Display Shape and Info vertically
                st.markdown(f"**Shape:** `{df_sidebar.shape[0]}` rows, `{df_sidebar.shape[1]}` columns")

                # Display Preview (Head)
                st.markdown("**Preview (First 5 Rows):**")
                st.dataframe(df_sidebar.head()) # Display the first 5 rows


# ==== APPLY CSS STYLES (Including Developer-Set Spacing) ====
# Inject CSS styles into the Streamlit app
st.markdown(f"""
<style>
    {font_style} /* Inject font style (Montserrat or fallback) */

    /* --- General Styles --- */
    .stApp {{
        background-color: #f8fbff; /* Light blue background */
    }}
    /* Main content container styling */
    .block-container {{
        padding: 0rem 3rem !important; /* Adjust left/right padding */
        max-width: 90% !important; /* Use full width */
    }}

    /* === Banner Image Styling === */
    .banner-container {{
        width: 100%;
        text-align: center;
        padding-top: 60px; /* Space above banner */
        margin-bottom: 10px; /* Space below banner */
    }}
    .banner-container img {{
        width: 50%; /* Adjust banner width */
        height: auto; /* Maintain aspect ratio */
        border-radius: 8px; /* Optional rounded corners */
    }}

    /* === Section Header (for main content sections) === */
    .section-header {{
        font-size: 18px;
        color: #112D4E; /* Dark blue text */
        margin-top: 0px;
        text-align: center;
        background-color: #9FB3DF; /* Light blue-grey background */
        margin-bottom: 10px;
        margin-bottom: 10px;
        font-weight: 600; /* Bold */
        padding: 3px;
        border-radius: 8px;
    }}

    /* === Column Wrapper for Developer-Set Spacing === */
    /* Applies right padding to all direct children (columns) except the last one */
    .column-wrapper > div {{
        padding-right: {DEVELOPER_COLUMN_SPACING_PX}px !important; /* Use developer-set spacing */
    }}
    .column-wrapper > div:last-child {{
        padding-right: 0px !important; /* No padding on the very last column */
    }}

    /* === Expander Styles (Generic) === */
    /* Style for expanders in the main content area (if any added later) */
    .main-content .streamlit-expanderHeader {{
        /* background-color: #e6f2ff; */ /* Example: different bg */
        border: none;
        border-radius: 5px;
        margin-bottom: 2px;
    }}
    /* Generic expander header text style */
     .streamlit-expanderHeader p {{
        font-size: 18px; /* Default size */
        color: #002244; /* Dark navy text */
        font-weight: 600;
        padding: 8px 12px;
     }}
    /* Generic expander content area style */
    .streamlit-expanderContent {{
        background-color: #ffffff; /* White background */
        font-size: 15px;
        color: #333; /* Dark grey text */
        border: 1px solid #e6f2ff; /* Light blue border */
        border-top: none; /* Remove top border to connect with header */
        border-radius: 0 0 5px 5px; /* Round bottom corners */
        padding: 15px;
    }}

    /* --- Sidebar Specific Styles --- */
    /* Style sidebar headers (e.g., the H3 for overview) */
    .stSidebar h2, .stSidebar h3 {{
        color: #003366; /* Dark blue */
        margin-bottom: 10px;
    }}
    /* Style the expander header *specifically* in the sidebar */
    .stSidebar .streamlit-expanderHeader {{
       /* background-color: #e0e0e0; */ /* Optional: different bg */
       border-radius: 5px 5px 0 0; /* Round top corners */
    }}
     /* Sidebar expander header text style */
     .stSidebar .streamlit-expanderHeader p {{
        font-size: 16px !important; /* Slightly smaller */
        font-weight: bold !important;
     }}
     /* Sidebar expander content area style */
     .stSidebar .streamlit-expanderContent {{
        background-color: #f0f5ff; /* Very light blue background */
        border: 1px solid #cce0ff; /* Slightly darker blue border */
        border-top: none;
        padding: 10px; /* Adjust padding */
     }}

    /* Uploader box style */
    .stSidebar .stFileUploader > div > div {{
        background-color: #FFF5E0 !important; /* Light yellow */
        border: 2px dashed #004080 !important; /* Dashed blue border */
        padding: 15px !important;
        border-radius: 8px !important;
    }}
    /* Uploader label (hidden via label_visibility, but kept for potential future use) */
    .stSidebar .stFileUploader label {{
        font-size: 16px !important;
        color: #003366 !important;
        font-weight: bold !important;
        margin-bottom: 10px;
    }}
     /* Browse files button style */
    .stSidebar .stFileUploader button {{
        background-color: #27548A !important; /* Dark blue */
        color: white !important;
        font-size: 14px !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        border: none !important;
        width: 100%; /* Full width */
        margin-top: 10px;
    }}
    .stSidebar .stFileUploader button:hover {{
        background-color: #112D4E !important; /* Darker blue on hover */
    }}
     /* Remove Dataset button style */
     /* Targets stButton within stSidebar for specificity */
    .stSidebar .stButton button {{
        background-color: #cc0000 !important; /* Red */
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 5px !important;
        width: 100%; /* Full width */
        margin-top: 15px; /* Space above */
        padding: 8px 16px !important;
    }}
     .stSidebar .stButton button:hover {{
        background-color: #aa0000 !important; /* Darker red on hover */
    }}
    /* Success message style in sidebar */
    .stSidebar .stSuccess {{
        font-size: 14px;
        padding: 10px;
        border-radius: 5px;
        /* Uses Streamlit's default success colors */
    }}
    /* Custom label style (used in markdown for uploader prompt) */
    .custom-label {{
        font-size: 16px; /* Adjust size */
        color: #333; /* Adjust color */
        margin-bottom: 5px;
        /* Font family is inherited from global styles */
    }}

</style>
""", unsafe_allow_html=True) # Render the CSS

# ==== DISPLAY BANNER IMAGE ====
# Display the banner image if it was loaded successfully
if img_base64:
    st.markdown(f"""
    <div class="banner-container" id="banner">
        <img src="data:image/png;base64,{img_base64}" class="banner-img" alt="Taal Lake Banner"/>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback text if the image failed to load
    st.markdown(
        "<div class='banner-container' id='banner' style='min-height: 80px; background-color: #eee; display: flex; align-items: center; justify-content: center;'><p style='color: #555;'>Banner Image Area (Image not found)</p></div>",
        unsafe_allow_html=True)

# ==== MAIN CONTENT AREA ====
st.markdown("<hr style='margin-top: 0; margin-bottom: 15px;'>", unsafe_allow_html=True) # Separator below banner

# Display analysis and prediction sections ONLY if a file is uploaded
if st.session_state["file_uploaded"] and st.session_state["df"] is not None:
    df = st.session_state["df"] # Get the dataframe for use in the main area

    # --- Analysis Section ---
    # Find numeric columns for plotting and correlation
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Check if any numeric columns were found
    if not numeric_cols:
        st.warning(
            "⚠️ No numeric columns found in the uploaded dataset. Trend visualization and correlation analysis require numeric data."
        )
    else:
        # Apply the column wrapper DIV to control spacing between analysis columns
        st.markdown('<div class="column-wrapper">', unsafe_allow_html=True)
        col_trend, col_corr = st.columns(2) # Create two columns in the main area

        # --- Trend Visualization Column ---
        with col_trend:
            with st.container(): # Group elements within the column
                st.markdown("<div class='section-header'>📈 Trend Visualization</div>", unsafe_allow_html=True)
                st.markdown("<div class='custom-label'>Select numeric columns to plot:</div>", unsafe_allow_html=True)

                # Pre-select first 2 numeric columns by default (if available)
                default_selection = numeric_cols[:min(len(numeric_cols), 2)]
                selected_cols = st.multiselect(
                    "Select columns:", # Label for the multiselect
                    numeric_cols,       # Options are the numeric columns
                    default=default_selection, # Default selection
                    key="trend_cols_inline",   # Unique key
                    label_visibility="collapsed" # Hide label, use markdown label above
                 )

                # Plot if columns are selected
                if selected_cols:
                    try:
                        # Create a DataFrame with only the selected columns
                        valid_cols_df = df[selected_cols].copy()
                        # Attempt to convert just in case (handles potential non-numeric issues)
                        for col in valid_cols_df.columns:
                             valid_cols_df[col] = pd.to_numeric(valid_cols_df[col], errors='coerce')
                        # Drop columns that became entirely NaN after coercion
                        valid_cols_df.dropna(axis=1, how='all', inplace=True)

                        # Plot only if there's data left
                        if not valid_cols_df.empty:
                             st.line_chart(valid_cols_df)
                        else:
                             st.warning("Selected columns contain non-numeric data or only missing values after conversion.")
                    except Exception as e:
                        st.error(f"Error plotting trends: {e}")
                elif numeric_cols: # Show info only if there were numeric cols to select
                    st.info("ℹ️ Select one or more numeric columns from the list above to see trends.")

        # --- Correlation Analysis Column ---
        with col_corr:
            with st.container(): # Group elements within the column
                st.markdown("<div class='section-header'>🔍 Correlation Analysis</div>", unsafe_allow_html=True)

                # Correlation requires at least 2 numeric columns
                if len(numeric_cols) >= 2:
                    try:
                        # Calculate correlation matrix on numeric columns only
                        corr = df[numeric_cols].corr()
                        # Create matplotlib figure and axes for the heatmap
                        fig, ax = plt.subplots(figsize=(6, 5)) # Adjust size as needed
                        sns.heatmap(
                            corr,
                            annot=True,      # Show correlation values
                            cmap='coolwarm', # Color map (blue-red)
                            fmt=".2f",       # Format values to 2 decimal places
                            linewidths=.5,   # Add lines between cells
                            ax=ax,           # Draw on the created axes
                            annot_kws={"size": 7} # Adjust annotation font size
                        )
                        plt.title("Correlation Matrix", fontsize=12)
                        plt.tight_layout() # Adjust plot to prevent labels overlapping
                        st.pyplot(fig) # Display the matplotlib figure in Streamlit
                    except Exception as e:
                        st.error(f"Error generating correlation heatmap: {e}")
                else:
                    # Message if not enough numeric columns for correlation
                    st.warning("⚠️ Correlation heatmap requires at least 2 numeric columns.")

        # Close the column wrapper DIV
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Prediction Section ---
    # Added 'main-content' class for potential distinct styling (optional)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔮 Model Prediction (Placeholder)</div>", unsafe_allow_html=True)
    # Use an expander for the prediction details/code
    with st.expander("Future Predictions", expanded=False): # Start collapsed
        st.markdown(
            "<div class='custom-label'>This section is intended for demonstrating model predictions based on the uploaded data.</div>",
            unsafe_allow_html=True)
        st.info("ℹ️ No predictive model is currently loaded or implemented in this demo.")
        # Example code block showing how model loading might look
        st.code("""
# Example Placeholder: Load and predict with a pre-trained model
#
# import joblib # Or another library like pickle, tensorflow, torch, etc.
#
# try:
#     # Load your trained model file
#     # model = joblib.load('your_trained_model.pkl')
#
#     # Select the features the model expects (must match training)
#     # feature_columns = ['feature1', 'numeric_feature2', 'categorical_encoded']
#     # X_predict = df[feature_columns] # Prepare data from the uploaded df
#
#     # Make predictions
#     # predictions = model.predict(X_predict)
#     # probabilities = model.predict_proba(X_predict) # If applicable (classifiers)
#
#     # Display results
#     # st.write("Model Predictions (Sample):")
#     # result_df = pd.DataFrame({'Prediction': predictions})
#     # st.dataframe(result_df.head())
#
# except FileNotFoundError:
#     st.warning("⚠️ Model file ('your_trained_model.pkl') not found.")
# except Exception as e:
#     st.error(f"❌ Error during model loading or prediction: {e}")
        """, language='python')
    # Close the 'main-content' div
    st.markdown("</div>", unsafe_allow_html=True)


else:
    # Message shown in the main area when no file is uploaded yet
    st.info("👈 Please upload a CSV dataset using the sidebar expander to begin the analysis.")
    st.markdown("---") # Add a separator

# Footer Section
st.markdown("---") # Final separator
# --- Add Date/Time and Location ---
now = datetime.datetime.now()
current_time_str = now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z") # Customize format if needed

st.caption(f"Taal Lake Dashboard | BULASO - DENNA - EJERCITADO - ESPINO - INCIONG | Updated: {current_time_str}")