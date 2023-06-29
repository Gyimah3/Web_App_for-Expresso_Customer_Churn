# # Importing required Library
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
from sklearn import preprocessing
from PIL import Image
# Setting up page configuration and directory path





st.set_page_config(page_title="Customer churn prediction App", page_icon="🛳️", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))


# Setting background image

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-color:black;
background-image:
radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 40px),
radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 30px),
radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 40px),
radial-gradient(rgba(255,255,255,.4), rgba(255,255,255,.1) 2px, transparent 30px);
background-size: 550px 550px, 350px 350px, 250px 250px, 150px 150px;
background-position: 0 0, 40px 60px, 130px 270px, 70px 100px;

}

</style>
'''
st.markdown(page_bg_img,unsafe_allow_html=True)



# Setting up logo

st.image('https://miro.medium.com/v2/resize:fit:786/format:webp/1*xT7y7u-DDssc3P_nT_qp2Q.png', width=500,caption=None, use_column_width=None, clamp=100, channels="RGB", output_format='JPEG')


# # Setting up Sidebar
social_acc = ['Data Field Description', 'EDA', 'About App']
social_acc_nav = st.sidebar.radio('**INFORMATION SECTION**', social_acc)

if social_acc_nav == 'Data Field Description':
     st.sidebar.markdown("""
    The table below gives a description on the variables required to make predictions.
    | Variable      | Definition:       |
    | :------------ |:--------------- |
    | FREQUENCE     | number of times the client has made an income |
    | TENURE        | duration in the network |
    | FREQUENCE_RECH| number of times the customer refilled |
    | MONTANT       | top-up amount   |
    | DATA_VOLUME   | number of connections|
    | ORANGE        | call to orange |
    | TIGO          | call to Tigo   |
    | ZONE1         | call to zones 1   |
    | ZONE2         | call to zones 2   |
    | ARPU_SEGMENT  | income over 90 days / 3 |
    | ON_NET        | inter expresso call |
    | REGULARITY    | number of times the client is active for 90 days   |
    | FREQ_TOP_PACK | number of times client has activated the top pack packages|
    | REVENUE       | monthly income of each client   |
    """)

elif social_acc_nav == 'EDA':
    st.sidebar.markdown("<h2 style='text-align: center;'> Exploratory Data Analysis </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''The exploratory data analysis of this project can be find in a Jupyter notebook from the link below''')
    st.sidebar.markdown("[Open Notebook](https://github.com/Gyimah3/Expresso_Customer_Churn_Prediction)")
elif social_acc_nav == 'About App':
    st.sidebar.markdown("<h2 style='text-align: center;'> Customer Churn prediction App </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("""
                        | Brief Introduction|
                        | :------------ |
                        This projet is based on a Zindi challenge for an African telecommunications company (Expresso)
                        that provides customers with airtime and mobile data bundles. The objective of this challenge
                        is to develop a machine learning model to predict the likelihood of each customer “churning,”
                        i.e. becoming inactive and not making any transactions for 90 days. This solution will help
                        this telecom company to better serve their customers by understanding which customers are at risk of leaving""")
    st.sidebar.markdown("")
    st.sidebar.markdown("[ Visit Github Repository for more information](https://github.com/Gyimah3/Expresso_Customer_Churn_Prediction)")
    st.sidebar.markdown("Dedicated to: mom❄️ and Sis Evelyn❄️.")
    st.sidebar.markdown("")
    
# Config & Setup
@st.cache(allow_output_mutation=True)
def Load_ml_items(relative_path):
    "Load ML items to reuse them"
    with open(relative_path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


loaded_object = Load_ml_items(r'ml_com.pkl')



pipeline_of_my_model = loaded_object["pipeline"]
num_cols = loaded_object['numeric_columns']
cat_cols = loaded_object['categorical_columns']



# Setting up variables for input data
@st.cache()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame(
        dict(
            TENURE=[],
            MONTANT=[],
            FREQUENCE_RECH=[],
            REVENUE=[],
            ARPU_SEGMENT=[],
            FREQUENCE=[],
            DATA_VOLUME=[],
            ON_NET=[],
            ORANGE=[],
            TIGO=[],
            ZONE1=[],
            ZONE2=[],
            REGULARITY=[],
            FREQ_TOP_PACK=[]
           
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

# setting Title for forms
st.markdown("<h2 style='text-align: center;'> Customer Churn Prediction </h2> ", unsafe_allow_html=True)
st.markdown("<h7 style='text-align: center;'> Fill in the details below and click on SUBMIT button to make a prediction </h7> ", unsafe_allow_html=True)


# Creating columns for for input data(forms)
left_col,right_col = st.columns(2)#[#20,20],gap="small")

# Developing forms to collect input data
with st.form(key="information", clear_on_submit=True):
    # Setting up input data for 1st column
    left_col.markdown("**FIRST COLUMN DATA**")
    TENURE= left_col.selectbox("duration in the network:",['K > 24 month','J 21-24 month','I 18-21 month','H 15-18 month','G 12-15 month','F 9-12 month',  'E 6-9 month', 'D 3-6 month'])
    MONTANT = left_col.number_input("top-up amount:",  min_value=0, max_value= 1000000000)
    FREQUENCE_RECH = left_col.number_input("number of times the customer refilled:", min_value=0, max_value=1000)
    REVENUE = left_col.number_input("monthly income of each client:", min_value=0, max_value=1000000000)
    ARPU_SEGMENT=left_col.number_input("income over 90 days / 3:",min_value=0, max_value=30000)
    FREQUENCE = left_col.number_input("number of times the client has made an income:", min_value=0, max_value=100)
    DATA_VOLUME = left_col.number_input("number of connections:",min_value=0, max_value=10000)
    

    
     # Setting up input data for 2nd column
    right_col.markdown("**SECOND COLUMN DATA**")
    ON_NET = right_col.number_input("inter expresso call:", min_value=0, max_value=10000)
    ORANGE = right_col.number_input("call to orange:", min_value=0, max_value=10000)
    TIGO = right_col.number_input("call to Tigo:", min_value=0, max_value=10000)
    ZONE1 = right_col.number_input("call to Zones 1:", min_value=0, max_value=10000)
    ZONE2 = right_col.number_input("call to Zones 2:", min_value=0, max_value=10000)
    REGULARITY = right_col.number_input("number of times the client is active for 90 day:", min_value=0, max_value=100)
    FREQ_TOP_PACK=right_col.number_input("number of times the client has activated the top pack packages:", min_value=0, max_value=100)

    submitted = st.form_submit_button(label="Submit")
    
if submitted:
    # Saving input data as csv after submission
    pd.read_csv(tmp_df_file).append(
        dict(
            
               
            
                TENURE=TENURE,
                MONTANT=MONTANT,
                FREQUENCE_RECH=FREQUENCE_RECH,
                REVENUE=REVENUE,
                ARPU_SEGMENT=ARPU_SEGMENT,
                FREQUENCE=FREQUENCE,
                DATA_VOLUME=DATA_VOLUME,
                ON_NET=ON_NET,
                ORANGE=ORANGE,
                TIGO=TIGO,
                ZONE1=ZONE1,
                ZONE2=ZONE2,
                REGULARITY=REGULARITY,
                FREQ_TOP_PACK=FREQ_TOP_PACK
           
            ),
            ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    st.balloons()
     

    df = pd.read_csv(tmp_df_file)
    df= df.copy()
   
        
    # Making Predictions
    # Passing data to pipeline to make prediction
    pred_output = pipeline_of_my_model.predict(df)
    prob_output = np.max(pipeline_of_my_model.predict_proba(df))
    
    # Interpleting prediction output for display
    X= pred_output[-1]
    if X == 1:
        explanation = 'Person will Churn'
    else: 
        explanation = "Person won't churn"
    output = explanation
    

    # Displaying prediction results
    st.markdown('''---''')
    st.markdown("<h4 style='text-align: center;'> Prediction Results </h4> ", unsafe_allow_html=True)
    st.success(f"Predicted Survival: {output}")
    st.success(f"Confidence Probability: {prob_output}")
    st.markdown('''---''')

    # Making expander to view all records
    expander = st.expander("See all records")
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['CHURN']= pred_output
        st.dataframe(df)

