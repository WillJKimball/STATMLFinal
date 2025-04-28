# Basketball Betting Algorithm

A machine learning model that predicts college basketball game outcomes using KenPom statistics.

## Features
- Uses KenPom efficiency metrics and advanced statistics
- XGBoost model for point spread prediction
- Home court advantage consideration
- Real-time game prediction capability

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up KenPom credentials in the code

## Usage
Run the main script:
```bash
python basketball_predictor.py
```

To predict a specific game:
```python
prediction = predict_game(
    team1="Team A",
    team2="Team B",
    home_team="team1"  # or "team2" or None for neutral site
)
```

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- kenpompy 