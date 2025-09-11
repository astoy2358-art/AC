document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const correctBtn = document.getElementById('correctBtn');
    const loading = document.getElementById('loading');
    const errorMsg = document.getElementById('errorMsg');

    correctBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();
        
        if (!text) {
            showError('Please enter some text to correct.');
            return;
        }
        
        // Show loading state
        loading.classList.remove('hidden');
        correctBtn.disabled = true;
        errorMsg.classList.add('hidden');
        
        try {
            const response = await fetch('/api/correct', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                outputText.innerHTML = `<p class="text-gray-800">${data.corrected_text}</p>`;
            } else {
                showError(data.error || 'An error occurred while processing your text.');
            }
        } catch (error) {
            showError('Network error. Please check your connection and try again.');
        } finally {
            loading.classList.add('hidden');
            correctBtn.disabled = false;
        }
    });
    
    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.classList.remove('hidden');
    }
});
