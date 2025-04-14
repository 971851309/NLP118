import streamlit as st
from agent import run_agent

# Set the page configuration.
st.set_page_config(page_title="CrewAI Sentiment Analysis", layout="centered")
st.title("CrewAI Sentiment Analysis and Response Generation")
st.markdown("Enter text below for sentiment analysis:")

# Manual input for customer name and purchase date
get_cust_name = st.text_input("Enter Customer Name")
get_purch_date = st.date_input("Enter Purchase Date", value=None, min_value=None, max_value=None, format="YYYY-MM-DD")

# List of products from the Amazon product list.
products = ['Apparel', 'Automotive', 'Baby', 'Beauty', 'Books', 'Camera',
       'Digital_Ebook_Purchase', 'Digital_Music_Purchase',
       'Digital_Software', 'Digital_Video_Download',
       'Digital_Video_Games', 'Electronics', 'Furniture', 'Gift Card',
       'Grocery', 'Health & Personal Care', 'Major Appliances',
       'Mobile_Apps', 'Mobile_Electronics', 'Musical Instruments',
       'Music', 'Office Products', 'Outdoors', 'PC',
       'Personal_Care_Appliances', 'Pet Products', 'Shoes', 'Software',
       'Sports', 'Tools', 'Toys', 'Video DVD', 'Video Games', 'Video',
       'Watches', 'Wireless']  # amazon product list
product = st.selectbox("Select Product Purchased", products)

# Text input for review
user_review = st.text_area("Your Review", "", height=150)

# Button to trigger analysis.
if st.button("Analyze"):
    # Validate that all required fields are filled and review has at least 1 word.
    if not get_cust_name or not get_purch_date or not product or len(user_review.split()) < 1:
        st.error("Please fill in all fields (Customer Name, Purchase Date, Product, and at least 1 word in the review).")
    else:
        # Combine all inputs into a single variable
        combined_input = f"name: {get_cust_name} purchase_date: {get_purch_date} product: {product} review: {user_review}"
        with st.spinner("Processing sentiment analysis for the review..."):
            result = run_agent(combined_input)
        
        st.success("Analysis complete!")
        
        # Display final reviewed response first (clearly visible, non-scrollable).
        st.markdown("### Final Reviewed Response")
        st.markdown(result['reviewed_response'])  # Fully displays the final reviewed output.
        
        # Place the remaining details within an expander (scrollable if needed).
        with st.expander("View Additional Details (Combined Input, Sentiment, Generated Response, Used Models, and Sentiment Review)", expanded=False):
            st.markdown("**Combined Input:**")
            st.write(combined_input)
            st.markdown("**Sentiment Analysis:**")
            st.write(result['sentiment'])
            st.markdown("**Generated Response:**")
            st.write(result['response'])
            st.markdown("**Models used:**")
            st.write(result['Used_Model'])
            st.markdown("**Sentiment Review:**")
            st.write(result['sentiment_review'])