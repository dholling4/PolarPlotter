import streamlit as st
import calendar
import pandas as pd
# Function to generate a calendar for a given year and month
def generate_calendar(year, month):
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]

    # Create a DataFrame to represent the calendar
    calendar_df = pd.DataFrame(cal, columns=[f"Week {i+1}" for i in range(len(cal[0]))])
    
    # Replace 0s with NaN for empty days
    calendar_df = calendar_df.replace(0, "")

    # Display the calendar DataFrame
    st.write(f"# Calendar for {month_name} {year}")
    st.dataframe(calendar_df)

st.markdown("# My Training Program")
st.sidebar.markdown("# My Training Program\U0001F4AA")



# Get user input for the year and month
year = st.number_input("Enter the year:", min_value=1, value=2024)
month = st.slider("Select the month:", 1, 12, 1)

# Generate and display the interactive calendar
#generatmulaton trainin

generate_calendar(year, month)
## 
