import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data
def load_data():
    df = pd.read_csv("process_2.csv")  # Ensure this file exists
    return df

# Sidebar Navigation
st.sidebar.title("📊 Banking Analytics Dashboard")
page = st.sidebar.radio("Select Page", ["Overview", "Exploratory Data Analysis", 
                                        "Customer Segmentation", "Credit Score Prediction",
                                        "Investment Recommendation"])

df = load_data()  # Load data once



### ------------------- PAGE 1: Overview -------------------
if page == "Overview":
    # Page title with icon
    st.title("🏦 Banking Data Analysis & Prediction")
    
    # Add a description for better context
    st.markdown("""
    Welcome to the **Banking Data Analysis & Prediction** dashboard. 
    This platform provides valuable insights into **credit risk assessment**, **customer segmentation**, and **recommendations** based on a dataset of **100,000 customer records**.  
    Use the sections below to explore the data and its key findings.
    """)

    # Add a radio button for navigation between sub-sections
    sub_page = st.radio("Select Section", ["Introduction", "Dataset Summary"])

    if sub_page == "Introduction":
        # Section Header
        st.header("📌 Introduction")
        
        # Content about banking and the dataset
        st.write("""
        In the banking sector, data plays a crucial role in enhancing decision-making, from loan approvals to credit scoring.
        This dashboard aims to uncover insights that can aid in improving **credit risk assessment**, better **customer segmentation**, and informed **investment strategies**.
        We are working with a dataset that contains **100,000 customer records**, offering a rich source of information for analysis and predictions.
        """)

    elif sub_page == "Dataset Summary":
        # Section Header
        st.header("📊 Dataset Overview")
        
        # Show a brief preview of the dataset
        st.subheader("Dataset Preview:")
        st.write("Below is a preview of the first few rows of the dataset:")
        st.dataframe(df.head())
        
        # Show basic statistics for the dataset
        st.subheader("Basic Statistics:")
        st.write("""
        Here are some basic statistics for the dataset, including summary metrics for numeric columns like mean, standard deviation, min, and max.
        """)
        st.write(df.describe())
        
        # If needed, add more information about the columns or missing data
        st.subheader("Dataset Insights:")
        st.write("""
        - Total records: 100,000
        - Number of features: 15 (e.g., age, income, credit score, loan status, etc.)
        - Check for missing values or anomalies to be handled in further analysis.
        """)

    # Optionally add a footer for extra details or credits
    st.markdown("""
    ---
    Report generated on: [19/2/2025]
    """)




### ------------------- PAGE 2: EDA -------------------
elif page == "Exploratory Data Analysis":
    # Page Title
    st.title("📈 Exploratory Data Analysis")
    
    # Introduction text to guide the user through the EDA process
    st.markdown("""
    In this section, we dive deep into the **Exploratory Data Analysis (EDA)** of the banking dataset.
    EDA is crucial for understanding patterns, correlations, and key insights from the data before moving on to more complex modeling.
    We will analyze various aspects of customer demographics, financial behavior, credit usage, and loan activity to identify potential opportunities and risks.
    Select a section below to begin exploring the different facets of the dataset.
    """)
    sub_page = st.selectbox(
        "Exploratory Data Analysis Sections",
        ["Customer Demographics & Distribution", "Income & Financial Behavior",
        "Credit Score Analysis", "Credit Utilization & Debt Patterns",
        "Loan & Credit Card Activity", "Payment & Spending Behavior"]
    )

    # Conditional rendering based on the selection
    if sub_page == "Customer Demographics & Distribution":
        st.title("Customer Demographics & Distribution")
        # Add your content or code for this section here
        st.write("""
        This section explores the distribution of various customer demographics, such as age, gender, and geographic information.
        Understanding customer demographics is key to segmentation and identifying trends across different customer profiles.
        Let's analyze the distribution of these key attributes to uncover patterns in the data.
        """)


        # Create the histogram with KDE
        fig = px.histogram(df, 
                        x="Age_Category", 
                        marginal="violin",  # Adds KDE estimation
                        color="Age_Category",
                        title="Age Category Distribution of Customers",
                        labels={"Age_Category": "Age Category"},
                        opacity=0.7,
                        color_discrete_sequence=["#1f77b4"] ) # Blue color for bars)  
        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Age Category",
            yaxis_title="Count",
            bargap=0.1,
            template="plotly_white"
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        # Group by Age_Category and sum the number of loans
        age_loan_data = df.groupby("Age_Category")["Num_of_Loan"].sum().reset_index()
        # Create a bar chart
        fig = px.bar(age_loan_data, 
                    x="Age_Category", 
                    y="Num_of_Loan", 
                    title="Total Number of Loans by Age Group",
                    labels={"Age_Category": "Age Group", "Num_of_Loan": "Total Loans"},
                    color="Num_of_Loan",
                    color_discrete_sequence=px.colors.sequential.Darkmint)
        # Display the plot
        st.plotly_chart(fig)



        # Count the number of loans per occupation
        loan_distribution = df.groupby("Occupation")["Num_of_Loan"].sum().reset_index()
        # Create a bar chart
        fig = px.bar(
            loan_distribution,
            x="Occupation",
            y="Num_of_Loan",
            title="Loan Distribution by Occupation",
            labels={"Num_of_Loan": "Number of Loans", "Occupation": "Occupation"},
            color="Num_of_Loan",
            color_discrete_sequence=px.colors.sequential.Darkmint
        )
        # Update layout for better readability
        fig.update_layout(xaxis_tickangle=-45)
        # Display the chart in Streamlit
        st.plotly_chart(fig)




        # Sample dataframe structure
        loan_columns = [
            "Auto Loan", "Mortgage Loan", "Debt Consolidation Loan", "Personal Loan",
            "Student Loan", "Payday Loan", "Home Equity Loan", "Credit-Builder Loan"
        ]
        # Ensure the columns for loan types exist in your dataframe
        loan_counts = df[loan_columns].apply(lambda x: (x == 1).sum(), axis=0).sort_values(ascending=False)
        # Create the bar chart
        fig = px.bar(
            x=loan_counts.index,
            y=loan_counts.values,
            title="Most Common Types of Loans",
            labels={"x": "Loan Type", "y": "Count"},
            color_discrete_sequence=["#1f77b4"]  # Blue color
        )
        # Update the chart to improve readability
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45)
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        loan_types = [
        "Auto Loan", "Mortgage Loan", "Debt Consolidation Loan",
        "Personal Loan", "Student Loan", "Payday Loan",
        "Home Equity Loan", "Credit-Builder Loan"
        ]
        # Aggregating loan data by Customer_Category
        loan_distribution = df.groupby("Customer_Category")[loan_types].sum().reset_index()
        # Melting the dataframe for Plotly
        loan_melted = loan_distribution.melt(id_vars=["Customer_Category"], 
                                            var_name="Loan Type", 
                                            value_name="Count")
        # Plot the grouped bar chart
        fig = px.bar(loan_melted, 
                    x="Customer_Category", 
                    y="Count", 
                    color="Loan Type", 
                    barmode="group",
                    title="Distribution of Loan Types Among Customer Categories",
                    labels={"Customer_Category": "Customer Category", "Count": "Number of Loans"},
                    color_discrete_sequence=["#1f77b4"]  # Blue color for bars
                    )

        fig.update_layout(xaxis_title="Customer Category", yaxis_title="Number of Loans")
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        # Count the number of loans each customer has by counting non-null/empty loan types
        df['Num_Loans'] = df['Type_of_Loan'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        # Count customers with multiple loans (more than 1 loan)
        multiple_loans_count = (df["Num_Loans"] > 1).sum()
        # Calculate percentage of customers with multiple loans
        total_customers = len(df)
        percentage_multiple_loans = (multiple_loans_count / total_customers) * 100
        # Create a pie chart for visualizing the results
        fig = px.pie(
            names=["Multiple Loans", "Single/No Loan"],
            values=[multiple_loans_count, total_customers - multiple_loans_count],
            title="Percentage of Customers with Multiple Loan Types",
            hole=0.4,
            color_discrete_sequence=["#1f77b4", "#e0e0e0"]  # Blue for multiple loans and light gray for single/no loan
        )
        # Streamlit visualization
        st.plotly_chart(fig)


    elif sub_page == "Income & Financial Behavior":
        st.title("Income & Financial Behavior")
        # Add your content or code for this section here
        st.write("""
        This section explores customers' income levels, spending behavior, and financial habits. 
        By analyzing this, we can identify correlations with credit usage, loan approvals, and potential financial risks.
        Let's investigate how income and financial behavior impact overall credit risk and loan activity.
        """)
        


        # Group by Income Category and calculate average Monthly Balance
        income_balance_avg = df.groupby("Income_Category")["Monthly_Balance"].mean().reset_index()
        # Create interactive bar chart
        fig = px.bar(
            income_balance_avg, 
            x="Income_Category", 
            y="Monthly_Balance", 
            title="Average Monthly Balance per Income Category",
            labels={"Monthly_Balance": "Avg Monthly Balance", "Income_Category": "Income Category"},
            color="Monthly_Balance",
            color_continuous_scale="Blues"
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        credit_score_balance = df.groupby('Credit_Score')['Monthly_Balance'].median().reset_index().round(1)
        # Plotting the median monthly balance across different credit scores
        fig = px.bar(credit_score_balance, 
                    x='Credit_Score', 
                    y='Monthly_Balance', 
                    title="Average Monthly Balance Across Credit Scores", 
                    labels={"Credit_Score": "Credit Score", "Monthly_Balance": "Median Monthly Balance"},
                    color='Monthly_Balance', 
                    color_discrete_sequence=['#1E3A8A']  # Blue color (Tailwind blue)

        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        # Calculate the average Monthly Balance for each Delayed Payment Category
        avg_balance_per_category = df.groupby('Delayed_Payment_Category')['Monthly_Balance'].mean().reset_index()
        # Create the bar plot with blue color
        fig = px.bar(avg_balance_per_category, 
                    x='Delayed_Payment_Category', 
                    y='Monthly_Balance', 
                    title='Average Monthly Balance by Delayed Payment Category',
                    labels={'Monthly_Balance': 'Average Monthly Balance', 'Delayed_Payment_Category': 'Delayed Payment Category'},
                    color='Monthly_Balance', 
                    color_discrete_sequence=px.colors.sequential.Darkmint)  # Use blue color palette
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        



        # Calculate the average EMI per month for each salary range
        avg_emi_by_salary = df.groupby('Salary_Range')['Total_EMI_per_month'].mean().reset_index()
        # Create the bar plot with blue color
        fig = px.bar(avg_emi_by_salary, 
                    x='Salary_Range', 
                    y='Total_EMI_per_month', 
                    title='Average Total EMI per Month by Monthly Inhand Salary Range',
                    labels={'Salary_Range': 'Monthly Inhand Salary Range', 'Total_EMI_per_month': 'Average Total EMI per Month'},
                    color='Total_EMI_per_month', 
                    color_discrete_sequence=['#1f77b4'])  # Blue color
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        # Calculate the average EMI per month for each salary range
        avg_emi_by_salary = df.groupby('Delayed_Payment_Category')['Outstanding_Debt'].mean().reset_index()
        # Create the bar plot
        fig = px.bar(avg_emi_by_salary, 
                    x='Delayed_Payment_Category', 
                    y='Outstanding_Debt', 
                    title='Average Outstanding Debt by Delayed Payment Category',
                    labels={'Delayed_Payment_Category': 'Delayed Payment Category', 'Outstanding_Debt': 'Average Outstanding Debt'},
                    color='Outstanding_Debt',
                    color_discrete_sequence=['#1f77b4'])  # Blue color
        # Show the plot in Streamlit
        st.plotly_chart(fig)



        # List of loan columns
        loan_columns = [
            "Auto Loan", "Mortgage Loan", "Debt Consolidation Loan", "Personal Loan",
            "Student Loan", "Payday Loan", "Home Equity Loan", "Credit-Builder Loan"
        ]
        # Calculate the average outstanding debt for each loan type
        avg_debt_per_loan_type = {}
        for loan_type in loan_columns:
            # Filter rows where the loan type is 1 (indicating the loan exists)
            loan_data = df[df[loan_type] == 1]
            # Calculate average outstanding debt for this loan type
            avg_debt_per_loan_type[loan_type] = loan_data['Outstanding_Debt'].mean()
        # Convert to DataFrame for plotting
        avg_debt_df = pd.DataFrame(list(avg_debt_per_loan_type.items()), columns=['Loan Type', 'Average Outstanding Debt'])
        # Plot using Plotly
        fig = px.bar(avg_debt_df, x='Loan Type', y='Average Outstanding Debt', 
                    title="Average Outstanding Debt per Loan Type", 
                    labels={'Loan Type': 'Loan Type', 'Average Outstanding Debt': 'Average Outstanding Debt'},
                    color='Average Outstanding Debt', 
                    color_discrete_sequence=['#1f77b4'])  # Blue color
        fig.update_layout(xaxis_tickangle=-45)
        # Display the plot in Streamlit
        st.plotly_chart(fig)




    elif sub_page == "Credit Score Analysis":
        st.title("Credit Score Analysis")
        # Add your content or code for this section here
        st.write("""
        The credit score is a critical factor in determining creditworthiness. In this section, we analyze the distribution of credit scores across customers and identify any trends or patterns.
        By correlating these scores with other features, we can gain insights into how different customer profiles impact credit risk.
        """)



        # Create the count plot with Plotly
        fig = px.histogram(df, x="Credit_Score", 
                        category_orders={"Credit_Score": df["Credit_Score"].value_counts().index},
                        color="Credit_Score", 
                        color_discrete_sequence=px.colors.sequential.Darkmint,  
                        title="Credit Score Distribution")
        # Customize the layout
        fig.update_layout(
            xaxis_title="Credit Score Category",
            yaxis_title="Number of Customers",
            xaxis_tickangle=45,
            font=dict(family="Arial", size=12, weight='bold')
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig)


        # Group by 'Age_Category' and 'Credit_Score' to get the count of each credit score per age category
        credit_score_by_age_category = df.groupby(['Age_Category', 'Credit_Score']).size().reset_index(name='Count')

        # Create a vertical bar plot using Plotly
        fig = px.bar(credit_score_by_age_category, 
                    x='Age_Category',  # Ensures bars are vertical
                    y='Count',         # Height of the bars
                    color='Credit_Score',
                    title='Credit Score Distribution by Age Category',
                    labels={'Count': 'Number of Individuals', 'Age_Category': 'Age Category', 'Credit_Score': 'Credit Score'},
                    color_discrete_sequence=px.colors.sequential.Darkmint  
                    )

        # Show the plot in Streamlit
        st.plotly_chart(fig)

            

        # Group by 'Spending_Level' and 'Credit_Score' to get the count of each credit score per spending level
        credit_score_by_age_category = df.groupby(['Spending_Level', 'Credit_Score']).size().reset_index(name='Count')
        # Create a bar plot using Plotly with blue color scheme
        fig = px.bar(credit_score_by_age_category, 
                    x='Spending_Level', 
                    y='Count', 
                    color='Credit_Score',
                    color_discrete_sequence=px.colors.sequential.Darkmint ,  # Blue color scale
                    title='Credit Score Distribution by Spending Level',
                    labels={'Count': 'Number of Individuals', 'Spending_Level': 'Spending Level', 'Credit_Score': 'Credit Score'},
                    barmode='stack')
        # Display the plot in Streamlit
        st.plotly_chart(fig)

        


    

        # Group by 'Payment_Value' and 'Credit_Score' to get the count of each credit score per payment value
        credit_score_by_payment_value = df.groupby(['Payment_Value', 'Credit_Score']).size().reset_index(name='Count')
        # Create a bar plot using Plotly
        fig = px.bar(credit_score_by_payment_value, 
                    x='Payment_Value', 
                    y='Count', 
                    color='Credit_Score',  # Color by Credit_Score, or any other field if needed
                    color_discrete_sequence=px.colors.sequential.Darkmint ,
                    title='Credit Score Distribution by Payment Value',
                    labels={'Count': 'Number of Individuals', 'Payment_Value': 'Payment Value', 'Credit_Score': 'Credit Score'},
                    barmode='stack')
        # Display the plot in Streamlit
        st.plotly_chart(fig)

        

        # Create a copy of the DataFrame
        df_copy = df.copy()
        # Encode categorical columns
        categorical_columns = df_copy.select_dtypes(include=['category', 'object']).columns
        # Label encode each categorical column
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
            label_encoders[col] = le
        # Calculate correlation matrix
        corr_matrix = df_copy.corr()
        # Get the correlation between 'Credit_Score' and other features
        credit_score_corr = corr_matrix['Credit_Score'].sort_values(ascending=False)
        # Create a bar plot for correlation with 'Credit_Score'
        plt.figure(figsize=(12, 6))
        credit_score_corr.plot(kind='bar', color='blue', title='Correlation with Credit Score')
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Display the plot in Streamlit
        st.pyplot(plt)



    elif sub_page == "Credit Utilization & Debt Patterns":
        st.title("Credit Utilization & Debt Patterns")
        # Add your content or code for this section here
        st.write("""
        This section focuses on analyzing how customers utilize credit, including credit card debt, outstanding loans, and overall credit usage patterns. 
        Understanding debt patterns can provide valuable insights into customers' financial health and predict potential credit risks.
        """)



        # Calculate correlation
        correlation = df['Credit_Utilization_Ratio'].corr(df['Outstanding_Debt'])
        # Display result with white color in Streamlit
        st.markdown(f"<h3 style='color:white;'>Correlation between Credit Utilization Ratio and Outstanding Debt: {correlation:.2f}</h3>", unsafe_allow_html=True)

        # Assuming df is already defined with necessary data
        loan_counts_by_month = df.groupby('Credit_Utilization_Ratio_Category')['Num_Credit_Card'].count().reset_index()
        # Creating a bar plot using Plotly with blue color
        fig = px.bar(loan_counts_by_month, 
                    x='Credit_Utilization_Ratio_Category', 
                    y='Num_Credit_Card', 
                    title='Number of Credit Cards on Credit Utilization Ratio Category', 
                    labels={'Credit_Utilization_Ratio_Category': 'Credit Utilization Ratio Category', 
                            'Num_Credit_Card': 'Number of Credit Cards'},
                    color='Num_Credit_Card',  # Keeps color based on category
                    color_discrete_sequence=['#1f77b4'])  # Blue color
        # Display the plot in Streamlit
        st.plotly_chart(fig)



        # Group the data by 'Type_of_Loan' and calculate the average outstanding debt
        outstanding_debt_by_loan_type = df.groupby('Type_of_Loan')['Outstanding_Debt'].mean().reset_index()
        # Create a bar chart to visualize the relationship
        fig = px.bar(outstanding_debt_by_loan_type,
                    x='Type_of_Loan', 
                    y='Outstanding_Debt',
                    title='Effect of Credit Card Type on Outstanding Debt',
                    labels={'Type_of_Loan': 'Type of Loan', 'Outstanding_Debt': 'Average Outstanding Debt'},
                    color='Outstanding_Debt',
                    color_continuous_scale='Blues')  # Blue color scale
        # Display the plot in Streamlit
        st.plotly_chart(fig)

        

        # List of loan columns
        loan_columns = [
            "Auto Loan", "Mortgage Loan", "Debt Consolidation Loan", "Personal Loan",
            "Student Loan", "Payday Loan", "Home Equity Loan", "Credit-Builder Loan"
        ]
        # Calculate the average outstanding debt for each loan type
        avg_debt_per_loan_type = {}
        for loan_type in loan_columns:
            # Filter rows where the loan type is 1 (indicating the loan exists)
            loan_data = df[df[loan_type] == 1]
            # Calculate average outstanding debt for this loan type
            avg_debt_per_loan_type[loan_type] = loan_data['Outstanding_Debt'].mean()
        # Convert to DataFrame for plotting
        avg_debt_df = pd.DataFrame(list(avg_debt_per_loan_type.items()), columns=['Loan Type', 'Average Outstanding Debt'])
        # Plot using Plotly
        fig = px.bar(avg_debt_df, x='Loan Type', y='Average Outstanding Debt', 
                    title="Average Outstanding Debt per Loan Type", 
                    labels={'Loan Type': 'Loan Type', 'Average Outstanding Debt': 'Average Outstanding Debt'},
                    color='Average Outstanding Debt', color_discrete_sequence=['#1f77b4'])  # Blue color
        fig.update_layout(xaxis_tickangle=-45)
        # Display the plot in Streamlit
        st.plotly_chart(fig)


    elif sub_page == "Loan & Credit Card Activity":
        st.title("Loan & Credit Card Activity")
        # Add your content or code for this section here
        st.write("""
        Here we explore customers' loan and credit card activity, including the frequency and amounts of loans taken or credit cards used.
        This analysis helps us identify trends in borrowing behavior, loan repayments, and credit card usage patterns.
        """)


        # Assuming 'df' is your DataFrame and it contains the necessary data
        loan_columns = ['Auto Loan', 'Mortgage Loan', 'Debt Consolidation Loan',
                        'Personal Loan', 'Student Loan', 'Payday Loan', 'Home Equity Loan',
                        'Credit-Builder Loan']
        # Filter rows where at least one loan column has a value of 1
        df_loans = df[df[loan_columns].any(axis=1)]  # Fix: Apply .any(axis=1)
        # Count occurrences of each loan type in each cluster
        loan_distribution = df_loans.groupby('Credit_Score')[loan_columns].sum().reset_index()
        # Melt the DataFrame to get a long format for better visualization
        loan_distribution_melted = loan_distribution.melt(id_vars=['Credit_Score'], 
                                                        var_name='Loan Type', 
                                                        value_name='Count')
        # Sort by 'Credit_Score' and 'Count', then get the top 5 loan types per cluster
        top_5_loans_per_cluster = loan_distribution_melted.sort_values(by=['Credit_Score', 'Count'], ascending=[True, False])
        # Create a stacked bar chart for the top 5 loans in each cluster (with vertical bars)
        fig = px.bar(top_5_loans_per_cluster, 
                    x='Loan Type',  # Loan Type on x-axis
                    y='Count',      # Count on y-axis
                    color='Credit_Score',  # Color by Credit_Score to differentiate clusters
                    title='Top 5 Loan Types Across Clusters', 
                    color_discrete_sequence=px.colors.sequential.Darkmint,  # Change to Viridis color scheme
                    barmode='stack')
        # Show the plot in Streamlit
        st.plotly_chart(fig)


        # Grouping by Income_Category and Age_Category, calculating the average number of loans
        avg_loans = df.groupby(["Income_Category", "Age_Category"])["Num_of_Loan"].mean().reset_index().round()
        # Creating the grouped bar chart
        fig = px.bar(avg_loans, 
                    x="Income_Category", 
                    y="Num_of_Loan", 
                    color="Age_Category",
                    barmode="group",
                    title="Average Number of Loans per Customer Across Income and Age Categories",
                    labels={"Num_of_Loan": "Average Number of Loans", 
                            "Income_Category": "Income Category", 
                            "Age_Category": "Age Category"},
                    color_discrete_sequence=px.colors.sequential.Darkmint)  # Blue color palette
        # Display the chart in Streamlit
        st.plotly_chart(fig)


        # Group the data and calculate average loans per income category
        df_grouped = df.groupby('Income_Category')['Num_of_Loan'].mean().reset_index()
        # Create the bar plot with blue color
        fig = px.bar(df_grouped, x='Income_Category', y='Num_of_Loan',
                    title='Average Number of Loans per Customer by Income Category',
                    labels={'Income_Category': 'Income Category', 'Num_of_Loan': 'Average Number of Loans'},
                    color='Income_Category', 
                    color_discrete_sequence=px.colors.sequential.Darkmint)  # Blue color
        # Display the plot in Streamlit
        st.plotly_chart(fig)


        loan_counts_by_month = df.groupby('Month')['Num_of_Loan'].sum().reset_index()
        # Creating a bar plot using Plotly with blue color
        fig = px.bar(loan_counts_by_month, x='Month', y='Num_of_Loan', 
                    title='Number of Loans Taken by Month', 
                    labels={'Num_of_Loan': 'Number of Loans', 'Month': 'Month'},
                    color='Month',
                    color_discrete_sequence=px.colors.sequential.Darkmint)  # Use blue color for the bars
        # Display the plot in Streamlit
        st.plotly_chart(fig)


    elif sub_page == "Payment & Spending Behavior":
        st.title("Payment & Spending Behavior")
        # Add your content or code for this section here
        st.write("""
        In this section, we examine how customers make payments and their spending patterns. 
        By analyzing transaction data and spending categories, we can uncover insights into customers' spending habits and predict future financial behavior.
        """)

        # Group by 'Payment_Value' and 'Credit_Score' to get the count of each credit score per age category
        credit_score_by_age_category = df.groupby(['Payment_Value', 'Credit_Score']).size().reset_index(name='Count')
        # Create a bar plot using Plotly
        fig = px.bar(credit_score_by_age_category, 
                    x='Payment_Value', 
                    y='Count', 
                    color='Credit_Score',
                    title='Payment Value Distribution by Spending Level',
                    labels={'Count': 'Number of Individuals', 'Payment_Value': 'Payment Value', 'Credit_Score': 'Credit Score'},
                    barmode='stack',
                    color_discrete_sequence=px.colors.sequential.Darkmint)  
        # Display the plot in Streamlit
        st.plotly_chart(fig)
            

        # Group by 'Changed_Credit_Limit_Category' and 'Credit_Score'
        credit_score_by_age_category = df.groupby(['Changed_Credit_Limit_Category', 'Credit_Score']).size().reset_index(name='Count')
        # Create a bar chart with a blue theme
        fig = px.bar(
            credit_score_by_age_category, 
            x='Changed_Credit_Limit_Category', 
            y='Count', 
            color='Credit_Score', 
            color_discrete_sequence=px.colors.sequential.Darkmint,  # Blue color
            title='Changed Credit Limit Distribution by Spending Level',
            labels={'Count': 'Number of Individuals', 'Changed_Credit_Limit_Category': 'Changed Credit Limit', 'Credit_Score': 'Credit Score'},
            barmode='stack'
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        # Group data
        credit_score_by_age_category = df.groupby(['Delay_Category', 'Credit_Score']).size().reset_index(name='Count')
        # Create a bar plot
        fig = px.bar(
            credit_score_by_age_category,
            x='Delay_Category',
            y='Count',
            color='Credit_Score',
            title='Delay Distribution by Spending Level',
            labels={'Count': 'Number of Individuals', 'Delay_Category': 'Delay', 'Credit_Score': 'Credit Score'},
            barmode='stack',
            color_discrete_sequence=px.colors.sequential.Darkmint  # Blue shades
        )
        # Display plot
        st.plotly_chart(fig, use_container_width=True)



### ------------------- PAGE 3: Customer Segmentation -------------------
elif page == "Customer Segmentation":
    st.title("🔄 Customer Segmentation Analysis")
    
    # Introduction Text
    st.markdown("""
    Welcome to the **Customer Segmentation** section. 
    In this analysis, we aim to group customers based on similar characteristics and behaviors. 
    By segmenting the customer base, we can uncover distinct customer profiles, which are crucial for targeted marketing, personalized offers, and improving customer experience.

    Through techniques such as clustering, we will identify meaningful segments that can help tailor strategies to meet the unique needs of each group. 
    Let's explore how segmenting the data can unlock new insights and drive better decision-making for the business.
    """)

    # Initialize the encoder dictionary to store mappings
    encoders = {}
    # Step 1: Encode categorical columns
    categorical_columns = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Value', 
                           'Credit_Score', 'Month', 'Payment_Behaviour', 'Customer_Category', 
                           'Spending_Level', 'Risk_Profile']
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder  # Store the encoder for reverse mapping
    # Step 2: Standardize the numerical features
    features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
                'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Outstanding_Debt',
                'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Changed_Credit_Limit',
                'Num_Credit_Inquiries', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    # Step 3: Apply KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    # Step 4: Decode categorical columns back to original values
    for col, encoder in encoders.items():
        df[col] = encoder.inverse_transform(df[col])
    # Rename clusters to segments
    cluster_names = {0: 'Segment-1', 1: 'Segment-2', 2: 'Segment-3'}
    df['Cluster'] = df['Cluster'].map(cluster_names)
    # Plot Cluster Distribution
    fig_pie = px.pie(df, names='Cluster', title="Cluster Distribution",
                      color='Cluster', color_discrete_sequence=px.colors.sequential.Darkmint)
    st.plotly_chart(fig_pie)
    # Dropdown to select segment
    segment_option = st.selectbox("Select a segment to visualize", list(cluster_names.values()))
    # Function to plot segment-wise Age Distribution
    def plot_segment(segment_name):
        segment_data = df[df['Cluster'] == segment_name]
        if 'Age_Category' in df.columns:
            fig = px.pie(segment_data, names='Age_Category', color='Age_Category',
                         title=f'Distribution of Age Category in {segment_name}', 
                         color_discrete_sequence=px.colors.sequential.Darkmint)
            return fig
        else:
            return None

    fig = plot_segment(segment_option)
    if fig:
        st.plotly_chart(fig)

    st.write("### **Recommendations**")
    st.write("""
        - **Early Career**: Entry-level credit products, salary-linked loans, and budgeting tools.
        - **Mid Career**: Larger loans, investment options, and family-oriented financial plans.
        - **Late Career**: Retirement savings, insurance, and wealth management.
        - **Senior**: Estate planning, retirement products, and healthcare financial solutions.
        - **Under-18**: Youth savings accounts and prepaid tools.
    """)

    # Total Annual Income per Segment
    income_sum_per_segment = df.groupby('Cluster')['Annual_Income'].sum().reset_index()
    fig_income = px.pie(income_sum_per_segment, names='Cluster', values='Annual_Income',
                        title='Total Annual Income by Segment',
                        color='Cluster', color_discrete_sequence=px.colors.sequential.Darkmint)
    st.plotly_chart(fig_income)

    st.write("### **Recommendations**")
    st.write("""
        - **Segment-1 (~$1.10B income)**: Budget-friendly financial products and low-interest credit cards.
        - **Segment-2 (~$1.15B income)**: Premium banking benefits, cashback rewards, and travel perks.
        - **Segment-3 (~$2.48B income)**: VIP banking, luxury credit cards, and real estate investments.
    """)

    # Occupation Distribution per Segment
    occupation_per_segment = df.groupby(['Cluster', 'Occupation']).size().reset_index(name='Count')
    # Select top 5 occupations for each segment
    top_5_occupation_per_segment = occupation_per_segment.groupby('Cluster').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)
    # Plot Stacked Bar Chart
    fig_occupation = px.bar(top_5_occupation_per_segment, x='Cluster', y='Count', color='Occupation',
                            title='Top 5 Occupations in Each Segment',
                            color_discrete_sequence=px.colors.sequential.Darkmint, barmode='stack')
    st.plotly_chart(fig_occupation)

    st.write("### **Recommendations**")
    st.write("""
        - **High-income professionals (Doctors, Lawyers, Entrepreneurs, Managers) **: should be offered premium banking, high-credit-limit cards, and tailored investment options.
        - **Self-employed & freelancers (Writers, Media Managers, Mechanics) **: need flexible credit products, business loans, financial literacy programs, business expansion loans, and digital payment solutions, along with travel and lifestyle banking benefits for media professionals.
        - **Accountants**: should receive tax-saving plans, structured investment products, and financial advisory services.
    """)

    # Group by Segment and Credit Score, count occurrences
    credit_score_distribution = df.groupby(['Cluster', 'Credit_Score']).size().reset_index(name='Count')

    # Create a stacked bar chart
    fig = px.bar(credit_score_distribution, 
                x='Cluster', 
                y='Count', 
                color='Credit_Score', 
                title='Distribution of Credit Score Categories Across Segments', 
                color_discrete_sequence=px.colors.sequential.Darkmint, 
                barmode='stack')

    st.plotly_chart(fig)

    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Poor Credit Score Customers**: Focus on debt restructuring, credit repair programs, and secured lending.
        - **Standard Credit Score Customers**: Offer low-risk financial products and incentives to improve credit score.
        - **Good Credit Score Customers**: Reward them with exclusive services like premium credit cards and wealth management.
    """)


    # Group by Cluster and calculate the average Credit Utilization Ratio
    avg_credit_utilization = df.groupby('Cluster')['Credit_Utilization_Ratio'].mean().reset_index()

    # Create a pie chart
    fig = px.pie(avg_credit_utilization, 
                names='Cluster', 
                values='Credit_Utilization_Ratio', 
                title='Average Credit Utilization Ratio per Cluster', 
                color_discrete_sequence=px.colors.sequential.Darkmint)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Cluster 0**:    31.64% (Moderate)
                - Likely consists of financially stable individuals who maintain a good balance between credit usage and available limits.

        - **Cluster 1**:    31.92% (Slightly Higher)
                - Similar to Cluster 0 but with a slightly higher utilization, possibly indicating occasional reliance on credit.

        - **Cluster 2**:    33.61% (Highest)
                - Customers in this group have the highest credit utilization, suggesting they may be more reliant on credit and at a higher risk of default if their financial situation worsens.
    """)


    # Group by Cluster and calculate the average Outstanding Debt
    debt_avg_per_cluster = df.groupby('Cluster')['Outstanding_Debt'].mean().reset_index()

    # Create a pie chart
    fig = px.pie(debt_avg_per_cluster, 
                names='Cluster', 
                values='Outstanding_Debt', 
                title='Average Outstanding Debt in Each Cluster', 
                color_discrete_sequence=px.colors.sequential.Darkmint)
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Cluster 0**:     High-Debt Customers
               - **Restrict High-Risk Lending**: Limit loans with high risks to reduce potential defaults.
               - **Encourage Debt Repayment**: Focus on programs to help customers manage and reduce existing debt.
        - **Cluster 1**:    Moderate-Debt Customers
                - **Expand Loan Offerings**: Provide more diverse loan options to meet their needs.
                - **Retain Loyal Customers**: Implement strategies to maintain long-term customer relationships.

        - **Cluster 2**:    Low-Debt Customers
                - **Encourage Smart Borrowing**: Promote responsible borrowing to avoid future financial strain.
                - **Promote Wealth-Building**: Offer products that support long-term financial growth and stability.
    """)


    # Group by Cluster and Age_Category, then calculate the mode of Num_of_Delayed_Payment
    delayed_payment_per_age = df.groupby(['Cluster', 'Age_Category'])['Num_of_Delayed_Payment'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]  # Handling mode and empty cases
    ).reset_index()

    # Create a bar plot to show the distribution of Num_of_Delayed_Payment by Age Category in each Cluster
    fig = px.bar(delayed_payment_per_age, 
                x='Cluster', 
                y='Num_of_Delayed_Payment', 
                color='Age_Category', 
                title='Most Frequent Number of Delayed Payments per Age Category in Each Cluster',
                barmode='group', 
                color_discrete_sequence=px.colors.sequential.Darkmint)

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Cluster 0**:     
                - Strict credit monitoring for all age groups, especially **Under 18** and **Early Career**. 
                - Implement stronger checks and balances to avoid defaults and encourage better credit management.

        - **Cluster 1**:    
                - Encourage **automatic payments** & **balance alerts** to prevent payment delays.
                - Provide financial tools to help maintain a consistent repayment schedule.
        
        - **Cluster 2**:   
                 - Monitor the **Under 18** category closely and introduce **early financial education programs**.
                 - Continue to promote healthy credit behaviors in this segment.
    """)



    # Group by Cluster and Age_Category, then calculate the average Monthly_Balance
    monthly_balance_per_age_cluster = df.groupby(['Cluster', 'Age_Category'])['Monthly_Balance'].sum().reset_index()
    # Create a bar plot to show the average Monthly Balance by Age Category in each Cluster
    fig = px.bar(monthly_balance_per_age_cluster, 
                x='Cluster', 
                y='Monthly_Balance', 
                color='Age_Category',
                title='Average Monthly Balance per Age Category in Each Cluster',
                barmode='group',
                color_discrete_sequence=px.colors.sequential.Darkmint)
    # Show the plot
    st.plotly_chart(fig)

    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Cluster 0**:     
                High monthly balances in Early Career (3.26M), Mid Career (2.87M), and Young (2.37M). Under 18 (1.11M) and Late Career (80K) have lower balances.

        - **Cluster 1**:    
                Highest balances in Early Career (3.99M), Mid Career (3.40M), and Late Career (2.60M). Seniors (76K) and Under-18 (376K) have significantly lower balances.
        
        - **Cluster 2**:   
                 Similar trend: Early Career (3.48M), Mid Career (3.25M), and Young (2.54M) have high balances. Seniors (64K) and Under-18 (285K) have the lowest balances.
    """)


    risk_profile_per_age_cluster = df.groupby(['Cluster', 'Age_Category', 'Risk_Profile']).size().reset_index(name='Count')
    # Bar plot for Risk Profile distribution per Age Category in each Cluster
    fig = px.bar(risk_profile_per_age_cluster, 
                x='Age_Category', 
                y='Count', 
                color='Risk_Profile', 
                facet_col='Cluster', 
                title='Risk Profile Distribution per Age Category in Each Cluster',
                barmode='stack',
                color_discrete_sequence=px.colors.sequential.Darkmint)

    st.plotly_chart(fig)

    
    # Recommendations section
    st.write("### **Recommendations**")
    st.write(""" 
        - **Cluster 0**:     
                High-Risk investors dominate, especially in Early Career, Mid Career, and Under 18. Low and Medium-Risk are much less frequent across all age groups.
        - **Cluster 1**:    
                High-Risk remains the largest group, especially in Early Career, Mid Career, and Young. Low-Risk investors are more frequent in Late Career and Senior.
        
        - **Cluster 2**:   
                 High-risk clients remain dominant across most age categories. Senior and Young groups show more Low-Risk preferences.
    """)



### ------------------- PAGE 4: Credit Score Prediction -------------------
elif page == "Credit Score Prediction":
    st.title("📊 Credit Score Prediction")
    st.write("🔹 **Credit Score Prediction**")
    st.write("""
        This tool helps users predict their **credit score** based on financial inputs such as 
        income, outstanding debt, loan details, and spending behavior.  
        
        ✅ **Enter your details**  
        ✅ **Get insights on your credit score**  
    """)
    
# # building model


    # Load Model
    model = pickle.load(open('my_model.pkl', 'rb'))

    # Streamlit UI
    st.title('Credit Score Prediction')

    # Create input form
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    age_category = st.selectbox("Age Category", ["Young Adults", "Adults", "Older Adults", "Middle-Aged Adults", "Teenagers"])
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", 
                                "July", "August", "September", "October", "November", "December"])

    occupation = st.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                                            'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
                                            'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])

    annual_income = st.number_input('Annual Income', min_value=100, max_value=10000000)
    income_category = st.selectbox("Income Category", ['Low Income', 'Lower Middle Income', 'High Income', 'Upper Middle Income'])

    monthly_inhand_salary = st.number_input('Monthly Inhand Salary', min_value=0, max_value=100000, step=100)
    salary_range = st.selectbox("Salary Range", ['700-900$', '1400-1500$', '1200-1400$', '300-500$', '500-700$', '900-1200$'])

    amount_invested_monthly = st.number_input('Amount Invested Monthly', min_value=0, max_value=100000, step=100)

    num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=1, max_value=20, step=1)
    num_credit_card = st.number_input('Number of Credit Cards', min_value=0, max_value=20, step=1)

    interest_rate = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, step=0.1)
    num_of_loan = st.number_input('Number of Loans', min_value=0, max_value=10, step=1)

    outstanding_debt = st.number_input('Outstanding Debt', min_value=0.0, max_value=1000000.0, step=1000.0)
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, max_value=1.0, step=0.01)

    credit_history_age_months = st.number_input('Credit History Age (Months)', min_value=0, max_value=1000, step=1)
    num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=500, step=1)

    num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, max_value=200, step=1)
    delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0, value=0)

    type_of_loan = st.selectbox("Type of Loan", ['Credit-Builder Loan', 'Auto Loan', 'Personal Loan',
                                                'Not Specified', 'Debt Consolidation Loan', 'Payday Loan',
                                                'Student Loan', 'Home Equity Loan', 'Mortgage Loan'])

    credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])

    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, step=500.0, value=1500.0)
    payment_of_min_amount = st.selectbox("Payment of Min Amount", ['No', 'Yes'])

    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=100.0, value=400.0)

    payment_behaviour = st.selectbox("Payment Behaviour", ['High_spent_Large_value_payments',
                                                        'Low_spent_Small_value_payments',
                                                        'High_spent_Medium_value_payments'])

    spending_level = st.selectbox("Spending Level", ['High', 'Low'])
    risk_profile = st.selectbox("Risk Profile", ['Low-Risk', 'High-Risk', 'Medium-Risk'])

    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Age_Category': age_category,
        'Month': month,
        'Occupation': occupation,
        'Annual_Income': annual_income,
        'Income_Category': income_category,
        'Monthly_Inhand_Salary': monthly_inhand_salary,
        'Salary_Range': salary_range,
        'Amount_invested_monthly': amount_invested_monthly,
        'Number_of_Bank_Accounts': num_bank_accounts,
        'Number_of_Credit_Cards': num_credit_card,
        'Interest_Rate': interest_rate,
        'Number_of_Loans': num_of_loan,
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization_ratio,
        'Credit_History_Age_Months': credit_history_age_months,
        'Number_of_Credit_Inquiries': num_credit_inquiries,
        'Number_of_Delayed_Payments': num_of_delayed_payment,
        'Delay_from_Due_Date': delay_from_due_date,
        'Type_of_Loan': type_of_loan,
        'Credit_Mix': credit_mix,
        'Changed_Credit_Limit': changed_credit_limit,
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_Month': total_emi_per_month,
        'Payment_Behaviour': payment_behaviour,
        'Spending_Level': spending_level,
        'Risk_Profile': risk_profile
    }])

    # Encoding categorical variables (Ensure they are preprocessed the same way as in training)
    categorical_cols = ['Age_Category', 'Month', 'Occupation', 'Income_Category', 'Salary_Range', 'Type_of_Loan',
                        'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Spending_Level', 'Risk_Profile']

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col])  # Ensure this matches training preprocessing
        label_encoders[col] = le  # Save encoder for future use

    btn = st.button("Submit")

    if btn:
        # Ensure the input data matches the expected training features
        expected_features = model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        result = model.predict(input_data)

        if result == 0:
            st.write("Poor Credit Score")
        elif result == 1:
            st.write("Standard Credit Score")
        elif result == 2:
            st.write("Good Credit Score")



### ------------------- PAGE 5: Loan Recommendation & Credit Score -------------------
elif page == "Investment Recommendation":
    
    df = pd.read_csv('Investment_recommendation.csv')
    
    # Load the trained model
    model = pickle.load(open('Investment_Recommendation.pkl', 'rb'))


    st.title("💼 Investment Recommendation System")

    # Create input form
    age = st.number_input('Age', min_value=18, max_value=100, step=1)
    age_category = st.selectbox("Age Category", ["Young Adults", "Adults", "Older Adults", "Middle-Aged Adults", "Teenagers"])
    month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", 
                                "July", "August", "September", "October", "November", "December"])

    occupation = st.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                                            'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
                                            'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])

    annual_income = st.number_input('Annual Income', min_value=100, max_value=10000000)
    income_category = st.selectbox("Income Category", ['Low Income', 'Lower Middle Income', 'High Income', 'Upper Middle Income'])

    monthly_inhand_salary = st.number_input('Monthly Inhand Salary', min_value=0, max_value=100000, step=100)
    salary_range = st.selectbox("Salary Range", ['700-900$', '1400-1500$', '1200-1400$', '300-500$', '500-700$', '900-1200$'])

    amount_invested_monthly = st.number_input('Amount Invested Monthly', min_value=0, max_value=100000, step=100)

    num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=1, max_value=20, step=1)
    num_credit_card = st.number_input('Number of Credit Cards', min_value=0, max_value=20, step=1)

    interest_rate = st.number_input('Interest Rate', min_value=0.0, max_value=100.0, step=0.1)
    num_of_loan = st.number_input('Number of Loans', min_value=0, max_value=10, step=1)

    outstanding_debt = st.number_input('Outstanding Debt', min_value=0.0, max_value=1000000.0, step=1000.0)
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0.0, max_value=1.0, step=0.01)

    credit_history_age_months = st.number_input('Credit History Age (Months)', min_value=0, max_value=1000, step=1)
    num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, max_value=500, step=1)

    num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, max_value=200, step=1)
    delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0, value=0)

    type_of_loan = st.selectbox("Type of Loan", ['Credit-Builder Loan', 'Auto Loan', 'Personal Loan',
                                                'Not Specified', 'Debt Consolidation Loan', 'Payday Loan',
                                                'Student Loan', 'Home Equity Loan', 'Mortgage Loan'])

    credit_mix = st.selectbox("Credit Mix", ['Good', 'Standard', 'Bad'])

    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, step=500.0, value=1500.0)
    payment_of_min_amount = st.selectbox("Payment of Min Amount", ['No', 'Yes'])

    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=100.0, value=400.0)

    payment_behaviour = st.selectbox("Payment Behaviour", ['High_spent_Large_value_payments',
                                                        'Low_spent_Small_value_payments',
                                                        'High_spent_Medium_value_payments'])

    spending_level = st.selectbox("Spending Level", ['High', 'Low'])
    risk_profile = st.selectbox("Risk Profile", ['Low-Risk', 'High-Risk', 'Medium-Risk'])
    Credit_Score = st.selectbox("Credit_Score", ['Good', 'Standard', 'Poor'])


    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Age_Category': age_category,
        'Month': month,
        'Occupation': occupation,
        'Annual_Income': annual_income,
        'Income_Category': income_category,
        'Monthly_Inhand_Salary': monthly_inhand_salary,
        'Salary_Range': salary_range,
        'Amount_invested_monthly': amount_invested_monthly,
        'Number_of_Bank_Accounts': num_bank_accounts,
        'Number_of_Credit_Cards': num_credit_card,
        'Interest_Rate': interest_rate,
        'Number_of_Loans': num_of_loan,
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization_ratio,
        'Credit_History_Age_Months': credit_history_age_months,
        'Number_of_Credit_Inquiries': num_credit_inquiries,
        'Number_of_Delayed_Payments': num_of_delayed_payment,
        'Delay_from_Due_Date': delay_from_due_date,
        'Type_of_Loan': type_of_loan,
        'Credit_Mix': credit_mix,
        'Changed_Credit_Limit': changed_credit_limit,
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_Month': total_emi_per_month,
        'Payment_Behaviour': payment_behaviour,
        'Spending_Level': spending_level,
        'Risk_Profile': risk_profile,
        'Credit_Score' : Credit_Score
    }])

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Convert all object type columns to numeric using LabelEncoder
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    btn = st.button("Submit")

    if btn:
        # Ensure the input data matches the expected training features
        expected_features = model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        result = model.predict(input_data)

        if result == 0:
            st.write("Invest 40% in low-risk assets, 30% in moderate-risk assets, 20% in high-risk stocks, and 10% in real estate.")
        elif result == 1:
            st.write("Prioritize debt repayment. Then, allocate 50% to low-risk investments, 30% to moderate-risk assets, and 20% to education or side business ventures.")
        elif result == 2:
            st.write("Invest 30% in low-risk, 40% in moderate-risk, 20% in high-risk stocks, and 10% in alternative investments like crypto.")
