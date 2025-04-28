import streamlit as st
from basketball_predictor import predict_game
from kenpompy.utils import login
from kenpompy import summary

def get_team_list():
    """Get list of all teams from KenPom"""
    browser = login('willkimball8@gmail.com', 'dkzTrGWm1G')
    stats_df = summary.get_efficiency(browser)
    return sorted(stats_df['Team'].unique().tolist())

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
    
    # Map venue selection to home_team parameter
    home_team_map = {
        "Team 1 Home": "team1",
        "Team 2 Home": "team2",
        "Neutral Site": None  # We'll handle this in predict_game
    }
    
    # Predict button
    if st.button('Predict Game'):
        with st.spinner('Calculating prediction...'):
            prediction = predict_game(
                team1=team1,
                team2=team2,
                home_team=home_team_map[venue]
            )
            
            st.subheader('Prediction Results')
            # Show who's favored
            if prediction > 0:
                favored_team = team1
                line = prediction
            else:
                favored_team = team2
                line = -prediction  # Make positive for display
                
            st.success(f"{favored_team} favored by {line:.1f} points")
            
            # Show betting line (from perspective of favored team)
            venue_text = "(neutral)" if venue == "Neutral Site" else ""
            st.write(f"Betting line {venue_text}: {favored_team} -{line:.1f}")

if __name__ == "__main__":
    main() 