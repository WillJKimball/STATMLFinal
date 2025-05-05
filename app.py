import streamlit as st
from basketball_predictor import get_prediction
from kenpompy.utils import login
from kenpompy import summary

def get_team_list():
    """Get list of all teams from KenPom"""
    try:
        browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
        stats_df = summary.get_efficiency(browser)
        return sorted(stats_df['Team'].str.lower().unique().tolist())
    except Exception as e:
        st.error(f"Error loading team list: {e}")
        return []

def main():
    st.title('College Basketball Predictor')
    
    # Get list of teams
    teams = get_team_list()
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Team 1')
        team1 = st.selectbox('Select Team 1', teams, key='team1')
        
    with col2:
        st.subheader('Team 2')
        team2 = st.selectbox('Select Team 2', teams, key='team2')
    
    # Venue selection
    venue = st.radio(
        "Select Venue",
        ["Team 1 Home", "Team 2 Home", "Neutral Site"],
        horizontal=True
    )
    
    # Predict button
    if st.button('Predict Game'):
        with st.spinner('Calculating prediction...'):
            try:
                # Get prediction result
                result = get_prediction(team1, team2, venue)
                
                # Round the line to the hundredths place
                result['line'] = round(result['line'], 2)
                
                # Display the result
                st.subheader('Prediction Results')
                st.success(f"{result['favored_team']} favored by {result['line']:.2f} points")
                
                # Show betting line
                venue_text = "(neutral)" if result['venue'] == "Neutral Site" else ""
                st.write(f"Betting line {venue_text}: {result['favored_team']} -{result['line']:.2f}")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()