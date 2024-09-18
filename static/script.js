const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = "black"; // Set canvas background to black
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white"; // Set stroke (digit) color to white
ctx.lineWidth = 10;
let drawing = false;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);

function startDrawing(e) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
}

function stopDrawing() {
    drawing = false;
}

document.getElementById('submit-btn').addEventListener('click', function() {
    const dataURL = canvas.toDataURL('image/png');
    const isEmpty = checkCanvasEmpty(); // Check if the canvas is empty

    if (isEmpty) {
        document.getElementById('prediction-result').innerText = 'Draw a digit between 0-9'; // Display message if canvas is empty
    } else {
    fetch('/predict-digit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction-result').innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error('Error:', error));
}
});

//To clear canvas
// Function to clear the canvas
document.getElementById('clear-button').addEventListener('click', function () {
    const canvas = document.getElementById('digit-canvas');
    const ctx = canvas.getContext('2d');
   // ctx.fillStyle = "black";
    
    // Clear the entire canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Optional: Reset the prediction result
    document.getElementById('prediction-result').innerText = '';
});

// Function to check if the canvas is empty (i.e., only contains the black background)
function checkCanvasEmpty() {
    const canvasData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = canvasData.data;

    // Iterate over pixel data in the canvas (RGBA values for each pixel)
    for (let i = 0; i < data.length; i += 4) {
        if (data[i] !== 0) { // Check the red component (you can check all RGB values for a more accurate check)
            return false; // If any pixel is not black, return false (canvas is not empty)
        }
    }
    return true; // If all pixels are black, return true (canvas is empty)
}