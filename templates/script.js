document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") sendMessage();
    });

    function sendMessage() {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;

        // Display user message
        const userMessageDiv = document.createElement("div");
        userMessageDiv.classList.add("message", "user-message");
        userMessageDiv.innerHTML = `<span>${userMessage}</span> <i class="fas fa-user icon"></i>`;
        chatBox.appendChild(userMessageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Send message to backend
        fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            const botMessageDiv = document.createElement("div");
            botMessageDiv.classList.add("message", "bot-message");
            botMessageDiv.innerHTML = `<i class="fas fa-robot icon"></i> <span>${data.response}</span>`;
            chatBox.appendChild(botMessageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            userInput.value = "";
        })
        .catch(error => console.error("Error communicating with server:", error));
    }
});
