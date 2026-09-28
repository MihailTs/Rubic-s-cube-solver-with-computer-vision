async function sendSolveRequest() {
    try {
        let object = {
            front: sharedCubeState.frontSide,
            top: sharedCubeState.topSide,
            left: sharedCubeState.leftSide,
            right: sharedCubeState.rightSide,
            bottom: sharedCubeState.bottomSide,
            back: sharedCubeState.backSide
        };

        object = Object.fromEntries(
            Object.entries(object).map(([key, element]) => [
                key,
                element.map(elem => sharedCubeState.number_to_color[elem])
            ])
        );

        const response = await fetch("http://localhost:8000/solve", {
            method: "POST",
            headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
            },
            body: JSON.stringify(object)
        });

        const data = await response.json();

        console.log("Server response:", data);

        const resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';

        if (response.status === 400 || data.error) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = data.error || 'An error occurred';
            resultContainer.appendChild(errorDiv);
        } else if (data.steps) {
            const steps = data.steps.trim().split(' ');
            const stepsContainer = document.createElement('div');
            stepsContainer.className = 'steps-container';
            
            steps.forEach(step => {
                const stepCard = document.createElement('div');
                stepCard.className = 'step-card';
                stepCard.textContent = step;
                stepsContainer.appendChild(stepCard);
            });
            
            resultContainer.appendChild(stepsContainer);
        }
    } catch (error) {
        const resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '';
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `Upload failed: ${error.message}`;
        resultContainer.appendChild(errorDiv);
    }
}