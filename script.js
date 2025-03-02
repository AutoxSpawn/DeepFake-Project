//JavaScript to handle user's input

const form = document.getElementById("news-form");
const loadingScreen = document.getElementById("loading-screen");
const resultScreen = document.getElementById("result-screen");
const resultText = document.getElementById("result-text");

form.addEventListener("submit", function(event) {
  event.preventDefault();
  
  // Show the loading screen
  loadingScreen.style.display = "block";
  resultScreen.style.display = "none";

  // Simulate the API request
  setTimeout(function() {
    loadingScreen.style.display = "none";
    resultScreen.style.display = "block";

    // Here, you would replace the static result with the actual response
    resultText.textContent = "This news is fake!";
  }, 2000); // Simulate a 2-second delay for loading
});
