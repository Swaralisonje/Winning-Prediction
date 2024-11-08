document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    const sport = document.getElementById('sport').value;

    // Make a POST request to the server with the sport data in JSON format
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',  // Set content type to JSON
        },
        body: JSON.stringify({ sport: sport })  // Send sport data as JSON
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Display the results in a table
        const resultContent = document.getElementById('result-content');
        resultContent.innerHTML = '';

        if (data.team_predictions) {
            let teamPredictions = data.team_predictions;

            let table = '<table><tr><th>Team</th><th>Medals Won</th><th>Chances of Winning (%)</th></tr>';
            let teamNames = [];
            let chancesOfWinning = [];

            teamPredictions.forEach(team => {
                table += `<tr><td>${team.Team}</td><td>${team.medals_won}</td><td>${(team.chances_of_winning * 100).toFixed(2)}%</td></tr>`;
                teamNames.push(team.Team);
                chancesOfWinning.push(team.chances_of_winning * 100);
            });
            table += '</table>';

            resultContent.innerHTML = `
                <p><strong>Sport:</strong> ${sport}</p>
                ${table}
            `;

            // Show chart section
            document.getElementById('chartSection').style.display = 'block';

            // Generate the chart
            createChancesOfWinningChart(teamNames, chancesOfWinning);
        } else {
            resultContent.innerHTML = '<p>No data available for the chosen sport.</p>';
            document.getElementById('chartSection').style.display = 'none';  // Hide chart if no data
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result-content').innerHTML = '<p>There was an error processing your request.</p>';
        document.getElementById('chartSection').style.display = 'none';  // Hide chart on error
    });
});

function createChancesOfWinningChart(teamNames, chancesOfWinning) {
    const ctx = document.getElementById('medalsChart').getContext('2d');

    // Check if the chart exists and is a Chart.js object, destroy it if so
    if (window.medalsChart instanceof Chart) {
        window.medalsChart.destroy();
    }

    // Create the new chart
    window.medalsChart = new Chart(ctx, {
        type: 'bar',  // You can change this to 'line', 'pie', etc.
        data: {
            labels: teamNames,  // X-axis labels
            datasets: [{
                label: 'Chances of Winning (%)',
                data: chancesOfWinning,  // Y-axis data (in percentage)
                backgroundColor: 'rgba(224, 255, 255, 0.8)',
                borderColor: 'rgba(0, 0, 139, 0.8)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 1)' // Change color of x-axis labels
                    }
                },
                y: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 1)', // Change color of y-axis labels
                        beginAtZero: true
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

