
document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault(); 
    const formData = {
        bedrooms: parseFloat(document.getElementById('bedrooms').value),
        bathrooms: parseFloat(document.getElementById('bathrooms').value),
        sqft_living: parseFloat(document.getElementById('sqft_living').value),
        sqft_lot: parseFloat(document.getElementById('sqft_lot').value),
        floors: parseFloat(document.getElementById('floors').value),
        waterfront: parseFloat(document.getElementById('waterfront').value),
        sqft_above: parseFloat(document.getElementById('sqft_above').value),
        sqft_basement: parseFloat(document.getElementById('sqft_basement').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('predictionResult').innerText = `$${data.prediction}`;
        } else {
            document.getElementById('predictionResult').innerText = `Error: ${data.error || 'Unknown error'}`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = 'An error occurred while fetching the prediction.';
    }
});
