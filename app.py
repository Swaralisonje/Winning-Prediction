from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the dataset
def load_data():
    filepath = r"C:\Users\Admin\Downloads\olympics_dataset.csv"  # Update to your file path
    return pd.read_csv(filepath)

# Preprocess the data
def preprocess_data(data):
    data['Medal_Won'] = data['Medal'].apply(lambda x: 0 if x == 'No medal' else 1)
    team_year_data = data.groupby(['Team', 'Year', 'Sport']).agg(
        total_participation=('player_id', 'count'),
        medals_won=('Medal_Won', 'sum')
    ).reset_index()

    team_year_data['win_percentage'] = (team_year_data['medals_won'] / team_year_data['total_participation']) * 100
    return team_year_data

# Plotting histogram function
def plot_histogram(sport_data, sport):
    plt.figure(figsize=(12, 6))
    filtered_data = sport_data[sport_data['win_percentage'] > 0]
    sorted_data = filtered_data.sort_values(by='win_percentage', ascending=False)
    sns.barplot(x='Team', y='win_percentage', data=sorted_data, palette='coolwarm')
    plt.title(f'Win Percentage by Country for {sport}', fontsize=16)
    plt.xlabel('Country (Team)', fontsize=12)
    plt.ylabel('Win Percentage', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    image_path = os.path.join('static', 'histogram.png')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

# Route for handling prediction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        sport = request.form.get("sport")
        data = load_data()
        team_year_data = preprocess_data(data)

        sport_data = team_year_data[team_year_data['Sport'] == sport]
        if sport_data.empty:
            return render_template('index.html', error=f"No data available for {sport}.")

        # Features and labels
        X = sport_data[['total_participation', 'medals_won']]
        sport_data['win_success'] = sport_data['win_percentage'].apply(lambda x: 1 if x > 50 else 0)
        y = sport_data['win_success']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle class imbalance using SMOTE for Naive Bayes
        sm = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

        # Train Naive Bayes on resampled data
        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train_resampled, y_train_resampled)
        y_pred_naive = naive_bayes.predict(X_test)
        naive_accuracy = accuracy_score(y_test, y_pred_naive)
        naive_conf_matrix = confusion_matrix(y_test, y_pred_naive)
        naive_precision = precision_score(y_test, y_pred_naive)
        naive_recall = recall_score(y_test, y_pred_naive)
        naive_f1 = f1_score(y_test, y_pred_naive)

        # Train Logistic Regression with class_weight balanced
        logistic_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
        logistic_reg.fit(X_train, y_train)
        y_pred_logistic = logistic_reg.predict(X_test)
        logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
        logistic_conf_matrix = confusion_matrix(y_test, y_pred_logistic)
        logistic_precision = precision_score(y_test, y_pred_logistic)
        logistic_recall = recall_score(y_test, y_pred_logistic)
        logistic_f1 = f1_score(y_test, y_pred_logistic)

        # Predicted winning chances for both models
        sport_data['winning_chance_naive_bayes'] = naive_bayes.predict_proba(X)[:, 1] * 100
        sport_data['winning_chance_logistic'] = logistic_reg.predict_proba(X)[:, 1] * 100

        # Filter countries with some winning chances
        sport_data = sport_data[(sport_data['winning_chance_naive_bayes'] > 0) |
                                (sport_data['winning_chance_logistic'] > 0)]

        if sport_data.empty:
            return render_template('index.html', error=f"No countries have a chance of winning in {sport}.")

        # Top 10 countries sorted by win_percentage
        top_10_countries = sport_data.sort_values(by='win_percentage', ascending=False).head(10)

        # Generate the histogram for all countries
        plot_histogram(sport_data, sport)

        # Determine the more accurate model
        more_accurate = "Logistic Regression" if logistic_accuracy > naive_accuracy else "Naive Bayes"

        # Convert top 10 results to a list of dictionaries
        results = top_10_countries[['Team', 'total_participation', 'medals_won', 'win_percentage']].to_dict(orient='records')
        for i, result in enumerate(results):
            result['winning_chance_logistic'] = round(top_10_countries['winning_chance_logistic'].iloc[i], 2)
            result['winning_chance_naive_bayes'] = round(top_10_countries['winning_chance_naive_bayes'].iloc[i], 2)

        # Render the template with all results
        return render_template('index.html',
                               sport=sport,
                               results=results,
                               logistic_accuracy=logistic_accuracy * 100,
                               naive_accuracy=naive_accuracy * 100,
                               more_accurate=more_accurate,
                               naive_conf_matrix=naive_conf_matrix,
                               logistic_conf_matrix=logistic_conf_matrix,
                               naive_precision=naive_precision,
                               naive_recall=naive_recall,
                               naive_f1=naive_f1,
                               logistic_precision=logistic_precision,
                               logistic_recall=logistic_recall,
                               logistic_f1=logistic_f1)

    # Default rendering if no sport is selected
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
