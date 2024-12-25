import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO
from methods import *
from RAG_pipeline import preprocess_data, create_vector_store, setup_rag_system
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Data Quality Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        if 'data' not in st.session_state:
            try:
                if uploaded_file.name.endswith(".csv"):
                   csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                   df = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                   df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.sidebar.success("Dataset uploaded successfully!")
            except Exception as e:
               st.sidebar.error(f"Error: {e}")
        else:
            df = st.session_state['data'].copy()


        if st.sidebar.button("Show Data", key='show_data_btn'):
            reset_all_flags()
            st.session_state['show_data'] = True

        if 'show_data' in st.session_state and st.session_state['show_data']:
            reset_all_flags()
            st.header("Data")
            st.write(df.head())

        # edited :addded
        if st.sidebar.button("Data Info", key='data_info_btn'):
            reset_all_flags()
            st.session_state['data_info'] = True

        if 'data_info' in st.session_state and st.session_state['data_info']:
            reset_all_flags()
            st.header("Data Info")
           # Capture the output of df.info() into a string
            buffer = StringIO()
            df.info(buf=buffer)
            info_string = buffer.getvalue()
    
           # Display the captured info in the Streamlit app
            st.text(info_string)

        if st.sidebar.button("Describe Data", key='describe_data_btn'):
             reset_all_flags()
             st.session_state['describe_data'] = True

        if 'describe_data' in st.session_state and st.session_state['describe_data']:
            reset_all_flags()
            st.header("Data Description")
            st.table(describe_data(df))

        if st.sidebar.button("Data Type Analysis", key='data_type_btn'):
            reset_all_flags()
            st.session_state['data_type_analysis_clicked'] = True
           # df = data_types_analysis(df)    Edited
        
        # Added the call of data_types_analysis
        if "data_type_analysis_clicked" in st.session_state and st.session_state["data_type_analysis_clicked"]:
            df = data_types_analysis(df)

        # Edited : Removed 
        # if 'type_converted' in st.session_state and st.session_state['type_converted']:
        #     st.write(df)
        #     reset_all_flags() # edited : Added
        #     st.session_state['type_converted'] = True # edited : false ==> true


        
        # if 'column_name_analysis_clicked' in st.session_state and st.session_state['column_name_analysis_clicked']:
        #     df = column_names_analysis(df)   
        #     st.session_state['column_name_analysis_clicked']= True # edited : false ==> true 

        # if 'columns_renamed' in st.session_state and st.session_state['columns_renamed']:
        #     reset_all_flags()
        #     #st.session_state['column_name_analysis_clicked']= True
        #     st.write(df)
        #     #st.session_state['columns_renamed'] = False
        #     #st.session_state['column_name_analysis_clicked']= False  
 
        if st.sidebar.button("Column Name Analysis", key='col_name_btn'):
            reset_all_flags()
            st.session_state['column_name_analysis_clicked'] = True

        if "column_name_analysis_clicked" in st.session_state and st.session_state["column_name_analysis_clicked"]:
            df = column_names_analysis(df)

        if "columns_renamed" in st.session_state and st.session_state["columns_renamed"]:
            st.write(df)
            reset_all_flags()  # Reset after showing the renamed columns


        if st.sidebar.button("Missing Value Analysis", key='missing_val_btn'):
            reset_all_flags()
            st.session_state['missing_analysis_run'] = True
            
        if 'missing_analysis_run' in st.session_state and st.session_state['missing_analysis_run']:
           st.header("Missing Value Analysis")
           missing_value_analysis(df)
           st.session_state['missing_analysis_run']= False

        method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
        column = st.sidebar.selectbox("Select Column (optional)", df.columns, key="missing_col")
        if st.sidebar.button("Handle Missing Values", key='handle_missing_btn'):
            reset_all_flags()
            df = handle_missing_values(df, method, column)
            st.session_state['data'] = df
            st.session_state['missing_values_handled'] = True

        if 'missing_values_handled' in st.session_state and st.session_state['missing_values_handled']:
            st.header("Data after Handling Missing Values")
            st.write(df)
            missing_value_analysis(df)
            st.session_state['missing_values_handled']= False

        # Edited : Added
        if st.sidebar.button("Show Duplicates", key='show_duplicates_btn'): 
            reset_all_flags()  # Ensure other flags are reset
            st.write("Checking for duplicate rows...")
            show_duplicates(df)



        #Sidebar button to handle duplicates
        if st.sidebar.button("Handle Duplicates", key='handle_duplicates_btn'):
            df = handle_duplicates(df)
            st.session_state['data'] = df

        # Display the DataFrame after handling duplicates
        if 'duplicates_handled' in st.session_state and st.session_state['duplicates_handled']:
        #if st.session_state.get('duplicates_handled', False):
            st.header("Data after Handling Duplicates")
            st.write(st.session_state['data'])
            
            # Reset the flag after displaying the updated DataFrame
            st.session_state['duplicates_handled'] = False


        

        # column_for_outlier = st.sidebar.selectbox("Select Column for Outlier Analysis", df.select_dtypes(include=['float64', 'int64']).columns, key="outlier_col")
        # if st.sidebar.button("Outlier Analysis", key='outlier_analysis_btn'):
        #      reset_all_flags()
        #      st.session_state['outlier_analysis_run'] = True

        # if 'outlier_analysis_run' in st.session_state and st.session_state['outlier_analysis_run']:
        #     st.header("Outlier Analysis")
        #     lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
        #     if lower_bound is not None and upper_bound is not None:
        #         outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method")
        #         if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
        #             df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
        #             st.session_state['data'] = df
        #             st.session_state['outliers_handled'] = True
        #             reset_all_flags()

        # if 'outliers_handled' in st.session_state and st.session_state['outliers_handled']:
        #      st.header("Data after Handling Outliers")
        #      st.write(df)
        #      st.session_state['outliers_handled']=False


        column_for_outlier = st.sidebar.selectbox(
        "Select Column for Outlier Analysis",
        df.select_dtypes(include=['float64', 'int64']).columns,
        key="outlier_col"
        )

        # Sidebar: Run outlier analysis
        if st.sidebar.button("Outlier Analysis", key='outlier_analysis_btn'):
            reset_all_flags()
            st.session_state['outlier_analysis_run'] = True

        # Perform outlier analysis
        if 'outlier_analysis_run' in st.session_state and st.session_state['outlier_analysis_run']:
            st.header("Outlier Analysis")
            lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)

            if lower_bound is not None and upper_bound is not None:
                # Sidebar: Select method to handle outliers
                outlier_method = st.sidebar.selectbox(
                    "Select Outlier Handling Method",
                    ['clip', 'drop'],
                    key="outlier_method"
                )

            # Sidebar: Handle outliers
            if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
                df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                st.session_state['data'] = df
                st.session_state['outliers_handled'] = True
                st.session_state['outlier_analysis_run'] = False  # Reset analysis flag

        # Display the DataFrame after handling outliers
        if st.session_state.get('outliers_handled', False):
            st.header("Data after Handling Outliers")
            st.write(st.session_state['data'])
            st.session_state['outliers_handled'] = False

        # Display the original DataFrame
        st.header("Original DataFrame")
        st.write(df)


        column_to_visualize = st.sidebar.selectbox("Select Column for Visualization", df.columns, key="visualize_col")
        if st.sidebar.button("Visualize Data", key='visualize_data_btn'):
            reset_all_flags()
            st.session_state['visualize_data_run'] = True

        if 'visualize_data_run' in st.session_state and st.session_state['visualize_data_run']:
            st.header("Data Visualization")
            fig1, fig2 = visualize_data(df, column_to_visualize)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.session_state['visualize_data_run'] = False

        if st.sidebar.button("Correlation Matrix", key='correlation_btn'):
            reset_all_flags()
            st.session_state['correlation_run'] = True

        if 'correlation_run' in st.session_state and st.session_state['correlation_run']:
            st.header("Correlation Matrix")
            fig = correlation_matrix(df)
            if fig is not None:
                st.pyplot(fig)
            st.session_state['correlation_run']= False

        if st.sidebar.button("Download dataset", key='download_btn'):
             download_dataset(df)

        # Upload dataset
        st.title("RAG-Powered Dataset Q&A")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

        if uploaded_file:
            # Preprocess data
            st.info("Processing dataset...")
            documents = preprocess_data(uploaded_file)
            st.success(f"Dataset loaded with {len(documents)} rows.")
            
            # Create vector store
            st.info("Creating vector store...")
            vector_store = create_vector_store(documents)
            st.success("Vector store created.")
            
            # Setup RAG
            st.info("Setting up RAG system...")
            qa_chain = setup_rag_system(vector_store)
            st.success("RAG system is ready.")
            
            # User Query
            query = st.text_input("Ask a question about your dataset:")
            if query:
                response = qa_chain.run(query)
                st.write("Answer:", response)   

            # import openai 

            # openai.api_key = "your-api-key"

            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[{"role": "user", "content": "Hello!"}],
            #     max_tokens=50,
            # )

            # print(response)
              

if __name__ == "__main__":
    main()