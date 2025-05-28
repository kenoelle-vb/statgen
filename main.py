# streamlit run "C:\Users\keno\OneDrive\Documents\Projects\statgen\0_FIXED.py"

import streamlit as st
import pandas as pd
import io 
import os 
import xlsxwriter
import requests
from PIL import Image
import base64
from PyPDF2 import PdfReader, errors
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import nltk
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


# ============================================== DESIGN UI ================================================================================


logo = Image.open("logo.png")

st.set_page_config(
    page_title="n_secondbrain",
    page_icon=logo,
    layout="wide",  # or "wide" if you prefer
    initial_sidebar_state="auto"
)

st.set_option('client.showErrorDetails', False)

st.markdown(
    """
    <style>
    section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
         padding-top: 0px;  # Remove padding completely
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit style elements
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="stTextArea"] {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Set Montserrat font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Change color of specific Streamlit elements
st.markdown("""
    <style>
    .st-emotion-cache-1o6s5t7 {
        color: #ababab !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stExpander {
        background-color: #FFFFFF;
        border-radius: 10px;
    }
    
    .stExpander > details {
        background-color: #FFFFFF;
        border-radius: 10px;
    }
    
    .stExpander > details > summary {
        background-color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        padding: 10px;
    }
    
    .stExpander > details > div {
        background-color: #FFFFFF;
        border-radius: 0 0 10px 10px;
        padding: 10px;
    }
    
    .stCheckbox {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .stButton > button {
        color: #FFFFFF;
        background-color: #424040;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        font-size: 20px;
    }
    .streamlit-expanderContent {
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_png_as_page_bg("background.jpg")


# ============================================== FUNCTIONS ===============================================================================


def extract_text_from_pdf_url_to_dataframe(url, chunk_size, chunk_overlap):
    """
    Downloads a PDF from a given URL, extracts its text, and splits it into chunks.
    Returns a pandas DataFrame where each row is a text chunk with its original URL.
    """
    try:
        # Download the PDF content
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Save the content to a temporary file
        with open('temp_pdf_for_extraction.pdf', 'wb') as f:
            f.write(response.content)

        # Extract text using PyPDF2
        reader = PdfReader('temp_pdf_for_extraction.pdf')
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Prepare for Langchain text splitting
        if not text.strip():
            print(f"Warning: No text extracted from PDF at {url}")
            return None

        # --- Modified Text Splitting ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[". ", ".\n", "\n\n", "\n", " ", ""] # Prioritize splitting at periods
        )

        chunks = text_splitter.split_text(text) # Use split_text for a raw string

        # Create a DataFrame from the chunks
        df = pd.DataFrame(columns=['Text', 'Link'])
        for chunk_text in chunks:
            # strip whitespace from chunks
            processed_chunk_text = chunk_text.strip()
            # Add preprocessing to remove leading '. '
            if processed_chunk_text.startswith('. '):
                processed_chunk_text = processed_chunk_text[2:] # Remove the first 2 characters ('. ')

            df.loc[len(df)] = [processed_chunk_text, url]

        return df

    except requests.exceptions.RequestException as e:
        print(f"Request Error downloading PDF from {url}: {e}")
        return None
    except errors.PdfReadError as e:
        print(f"PDF Read Error processing {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")
        return None

def preprocess_text(text):
    """
    Cleans and normalizes text for similarity comparison.
    Steps: lowercase, tokenize, remove non-alpha, remove stopwords, lemmatize.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def initialize_vectorizer_and_vectors(df):
    """
    Initializes a TfidfVectorizer and transforms the 'Text' column of the DataFrame
    into TF-IDF vectors.
    Returns the fitted vectorizer and the TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer()
    # Apply preprocessing to the 'Text' column
    df['preprocessed_text'] = df['Text'].apply(preprocess_text)
    vectors = vectorizer.fit_transform(df['preprocessed_text'])
    return vectorizer, vectors

def similarity_search(query, df, vectorizer, vectors, top_n=1, is_stat="YES", filter_strings=None):
    """
    Performs a similarity search on the text chunks using TF-IDF and cosine similarity.
    Applies optional filters for 'is_stat' (presence of digits) and 'filter_strings'.
    """
    # Preprocess the query and transform it into a TF-IDF vector
    query_vector = vectorizer.transform([preprocess_text(query)])

    # Calculate cosine similarities between the query and all document vectors
    similarities = cosine_similarity(query_vector, vectors).flatten()

    # Get indices of top N most similar chunks
    # We sort all similarities and take the top N. Filtering happens next.
    all_sorted_indices = similarities.argsort()[::-1] # Sort descending

    filtered_indices = []
    # Iterate through all sorted indices to find suitable chunks up to top_n
    for index in all_sorted_indices:
        if len(filtered_indices) >= top_n:
            break # Stop if we have enough results

        chunk_text = df.iloc[index]['Text']
        keep_chunk = True # Assume we keep the chunk initially

        # Apply is_stat filter
        if is_stat.upper() == "YES":
            if not any(char.isdigit() for char in chunk_text):
                keep_chunk = False # If is_stat is YES and no digit, do not keep

        # Apply filter_strings filter if specified and we are still considering keeping the chunk
        if keep_chunk and filter_strings: # Check if filter_strings is not None and not empty
            if isinstance(filter_strings, str):
                filter_list = [filter_strings]
            else:
                filter_list = filter_strings

            all_filter_strings_found = True # Assume all are found initially
            for filter_str in filter_list:
                if filter_str not in chunk_text:
                    all_filter_strings_found = False
                    break

            if not all_filter_strings_found:
                keep_chunk = False

        # If the chunk passed all filters, add its index
        if keep_chunk:
            filtered_indices.append(index)

    # Get the top texts and their links from the filtered list
    results = df.iloc[filtered_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    results = results[['Text', 'Link']]

    # Filter out duplicate texts (optional, but good practice for display)
    results.drop_duplicates(subset=['Text'], inplace=True)

    return results # Return a DataFrame of results


# =========================================== STREAMLIT APP CODE =================================================================================

colmain1, colmain2 = st.columns([1, 12])
with colmain1:
    image = Image.open('logo.png')
    resized_image = image.resize((80, 80))
    st.image(resized_image)
with colmain2:
    st.markdown(
            """
            <span style='color:white; font-size:36px; font-weight:bold; margin-top: 15px; display: block;'>statgen</span>
            """, 
            unsafe_allow_html=True
        )

# --- Initialize session state variables ---
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'filter_queries' not in st.session_state:
    st.session_state.filter_queries = [""] # Start with one empty input field
if 'searched_df' not in st.session_state:
    st.session_state.searched_df = None
if 'log_messages' not in st.session_state: # New session state for logs
    st.session_state.log_messages = []

# Function to add logs to session state
def add_log(message):
    st.session_state.log_messages.append(message)

# Clear logs on initial load or file upload to avoid carrying logs from previous runs
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None

### 1. Upload Excel File & Configure Chunking

# Expander 1: File Upload
with st.expander("", expanded=True):
    excel_file = st.file_uploader("Upload your Excel file (MUST BE .xlsx)", type=["xlsx"])

    df_input = None
    if excel_file:
        # Check if a new file has been uploaded
        if st.session_state.current_file_hash != excel_file.file_id:
            st.session_state.current_file_hash = excel_file.file_id
            # Clear all relevant session states for a fresh start with a new file
            st.session_state.combined_df = None
            st.session_state.vectorizer = None
            st.session_state.vectors = None
            st.session_state.searched_df = None
            st.session_state.log_messages = [] # Clear logs for new file

        try:
            df_input = pd.read_excel(excel_file)
            st.write("#### Excel File Preview:")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            df_input = None # Ensure df_input is None if there's an error

    else: # If no file is uploaded, reset everything
        st.session_state.combined_df = None
        st.session_state.vectorizer = None
        st.session_state.vectors = None
        st.session_state.searched_df = None
        st.session_state.log_messages = []
        st.info("Please upload an Excel file to proceed.")
        df_input = None # Explicitly set to None


if df_input is not None:
    # Expander 2: Link Column & Chunk Size
    with st.expander("", expanded=True):
        col_link_select, col_chunk_size = st.columns([0.30, 0.70])

        with col_link_select:
            column_name_for_links = st.selectbox(
                "Select the column containing PDF links:",
                options=df_input.columns.tolist(),
                index=df_input.columns.get_loc('Link') if 'Link' in df_input.columns else 0,
                key="link_column_select"
            )

        with col_chunk_size:
            chunk_size = st.select_slider(
                "Select Chunk Size:",
                options=[100, 150, 200, 250, 300, 350, 400, 450, 500, 
                         550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 
                         1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 
                         1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000],
                value=250,
                key="chunk_size_slider"
            )
            chunk_overlap = chunk_size // 4
            st.info(f"Chunk Overlap will be: **{chunk_overlap}**")

        # This button is in its own column for layout
        col_chunk_button_spacer, col_chunk_button = st.columns([0.80, 0.20])
        with col_chunk_button:
            if st.button("Run Chunking Sequence", key="run_chunking_button"):
                if column_name_for_links:
                    pdf_urls_to_process = df_input[column_name_for_links].dropna().astype(str).tolist()
                    if not pdf_urls_to_process:
                        add_log("Warning: No valid URLs found in the selected column.")
                        st.warning("No valid URLs found in the selected column.")
                    else:
                        st.session_state.log_messages = [] # Clear logs before a new run
                        add_log(f"Starting to process **{len(pdf_urls_to_process)}** PDF URLs...")
                        all_extracted_dfs = []
                        progress_bar = st.progress(0)
                        
                        # Create a specific container for live logs within the expander
                        log_display_container = st.container()

                        for i, url in enumerate(pdf_urls_to_process):
                            # Ensure the raw URL processing message also goes into the container
                            df_single_pdf = extract_text_from_pdf_url_to_dataframe(url, chunk_size, chunk_overlap)
                            if df_single_pdf is not None:
                                all_extracted_dfs.append(df_single_pdf)
                                with log_display_container:
                                    add_log(f"Successfully extracted from: {url}")
                            else:
                                with log_display_container:
                                    add_log(f"Skipping {url} due to extraction error.")
                            progress_bar.progress((i + 1) / len(pdf_urls_to_process))

                        if all_extracted_dfs:
                            st.session_state.combined_df = pd.concat(all_extracted_dfs, ignore_index=True)
                            st.success("PDFs successfully extracted and chunked!")

                            # Initialize vectorizer and vectors immediately after chunking
                            st.session_state.vectorizer, st.session_state.vectors = \
                                initialize_vectorizer_and_vectors(st.session_state.combined_df.copy())
                            st.success("Text vectorized for similarity search!")

                            # Clean up the temporary PDF file if it exists
                            if os.path.exists('temp_pdf_for_extraction.pdf'):
                                os.remove('temp_pdf_for_extraction.pdf')
                        else:
                            st.error("No content was successfully extracted from any of the provided URLs.")
                else:
                    st.warning("Please select a column containing PDF links.")

# --- Display Logs in a collapsible expander (outside the main config expander) ---
if st.session_state.log_messages:
    with st.expander("Show Processing Logs", expanded=False):
        for log_msg in st.session_state.log_messages:
            st.markdown(log_msg)

# --- Display Chunked DataFrame (OUT OF COLUMNS) ---
if st.session_state.combined_df is not None:
    st.markdown(
        """
        <span style='color:white; font-size:36px; font-weight:bold; margin-top: 15px; display: block;'>Chunked DataFrame</span>
        """, 
        unsafe_allow_html=True
    )
    st.dataframe(st.session_state.combined_df)

    # Download button for the chunked dataframe
    col_chunked_df_spacer, col_download_chunked = st.columns([0.80, 0.20])
    with col_chunked_df_spacer:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            st.session_state.combined_df.to_excel(writer, index=False, sheet_name='Chunked Data')
        output.seek(0)
        st.download_button(
            label="Download Chunked Data",
            data=output.getvalue(),
            file_name="chunked_pdf_content.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the entire chunked DataFrame as an XLSX file."
        )

# --- Separator ---
st.markdown("---")

### 2. Search & Filter Chunks

if st.session_state.combined_df is not None:
    with st.expander("", expanded=True):
        search_query = st.text_input("Enter your search query:", key="search_query_input")

    # Expander 3: Search Parameters
    with st.expander("", expanded=True):
        col_top_n, col_is_stat, col_add_filter_button = st.columns([0.50, 0.25, 0.25])
        with col_top_n:
            top_n_results = st.slider(
                "Number of Top Results (top_n):",
                min_value=1, max_value=50, value=10, key="top_n_slider"
            )
        with col_is_stat:
            st.markdown("Statistics Only/No?") # Visual separator to align with text input
            is_stat_toggle = st.toggle("Require Statistics (Digits)?", value=False, key="is_stat_toggle")
            is_stat_param = "YES" if is_stat_toggle else "NO"
        with col_add_filter_button:
            st.markdown("Keyword Filters?") # Visual separator to align with text input
            if st.button("Add New Keyword Filter", key="add_filter_button"):
                st.session_state.filter_queries.append("")

    # Display filter query input fields (outside the "Search Parameters" expander but linked logically)
    with st.expander("", expanded=True):
        st.write("#### Keyword Filters :")
        actual_filter_strings = []
        for i, query_val in enumerate(st.session_state.filter_queries):
            col_filter_input, col_filter_remove = st.columns([0.7, 0.3])
            with col_filter_input:
                st.session_state.filter_queries[i] = st.text_input(
                    f"Keyword {i+1}:",
                    value=query_val,
                    key=f"filter_query_{i}"
                )
            with col_filter_remove:
                # Only allow removing if there's more than one filter input field
                if len(st.session_state.filter_queries) > 1:
                    if st.button(f"Remove Keyword {i+1}", key=f"remove_filter_{i}"):
                        st.session_state.filter_queries.pop(i)
                        st.rerun() # Rerun to remove the widget

            if st.session_state.filter_queries[i].strip():
                actual_filter_strings.append(st.session_state.filter_queries[i].strip())

        filter_strings_param = actual_filter_strings if actual_filter_strings else None

        # Perform search button
        if st.button("Perform Search", key="perform_search_button"):
            if search_query and st.session_state.vectorizer is not None:
                with st.spinner("Searching for relevant chunks..."):
                    st.session_state.searched_df = similarity_search(
                        search_query,
                        st.session_state.combined_df,
                        st.session_state.vectorizer,
                        st.session_state.vectors,
                        top_n=top_n_results,
                        is_stat=is_stat_param,
                        filter_strings=filter_strings_param
                    )
                if st.session_state.searched_df is not None and not st.session_state.searched_df.empty:
                    st.success("Search complete!")
                    st.write("### Top Search Results:")
                    st.dataframe(st.session_state.searched_df)
                else:
                    st.warning("No results found matching your query and filters.")
            else:
                st.warning("Please enter a search query and ensure chunking was successful.")

    # Display and download searched DataFrame
    if st.session_state.searched_df is not None and not st.session_state.searched_df.empty:
        col_searched_df_spacer, col_download_searched = st.columns([0.80, 0.20])
        with col_download_searched:
            output_searched = io.BytesIO()
            with pd.ExcelWriter(output_searched, engine='xlsxwriter') as writer:
                st.session_state.searched_df.to_excel(writer, index=False, sheet_name='Search Results')
            output_searched.seek(0)
            st.download_button(
                label="Download Searched Results",
                data=output_searched.getvalue(),
                file_name="searched_pdf_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the current search results as an XLSX file."
            )
