document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Ambil data form
    const inputData = [
        parseFloat(document.getElementById('bedrooms').value),
        parseFloat(document.getElementById('bathrooms').value),
        parseFloat(document.getElementById('sqft_living').value),
        parseFloat(document.getElementById('sqft_lot').value),
        parseFloat(document.getElementById('floors').value),
        parseFloat(document.getElementById('waterfront').value),
        parseFloat(document.getElementById('sqft_above').value),
        parseFloat(document.getElementById('sqft_basement').value)
    ];

    try {
        const response = await fetch('https://Delvinsss-daaa.hf.space/run/predict', {  // ← Ganti ke /run/predict
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: inputData }),  // Gradio expects {data: [values]}
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log(result);
        
        // Gradio biasanya balasannya { data: [result] }
        document.getElementById('predictionResult').innerText = `$${result.data[0]}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = 'An error occurred while fetching the prediction.';
    }
});
