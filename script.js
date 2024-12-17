document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');

    if (response.ok) {
        const data = await response.json();
        if (data.prediction) {
            resultDiv.innerHTML = `<p>Результат: ${data.prediction}</p>`;
            errorDiv.innerHTML = '';
        } else {
            errorDiv.innerHTML = `<p>Помилка: Немає результату</p>`;
        }
    } else {
        errorDiv.innerHTML = `<p>Помилка сервера</p>`;
    }
});
