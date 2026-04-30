import React, { useState, useEffect, useRef } from "react";
import { FaMicrophone, FaPaperPlane } from "react-icons/fa";
import "./ChatbotPage.css";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { text: input, type: "user" }]);
    setInput("");
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:5001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      console.log("Response:", data);

      if (data.response) {
        const formattedResponse = (
          <div>
            {data.response.split("\n").map((line, index) => (
              <p key={index}>--{line}</p>
            ))}
          </div>
        );

        setMessages((prev) => [
          ...prev,
          { text: formattedResponse, type: "bot" },
        ]);
      } else {
        setError("⚠️ No valid response received from server.");
      }
    } catch (err) {
      console.error("Error:", err);
      setError("❌ Server error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => e.key === "Enter" && handleSend();

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedImage(file);
    setMessages((prev) => [
      ...prev,
      { image: URL.createObjectURL(file), type: "user" },
    ]);
    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", file);
      const response = await fetch("http://127.0.0.1:5001/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("Image Response:", data);

      if (data.Options && Array.isArray(data.Options)) {
        const formattedPredictions = (
          <div>
            <p>
              🩺 <strong>Diagnosis Results:</strong>
            </p>
            <ul>
              {data.Options.map((option, index) => (
                <li key={index}>
                  🔹 <strong>{option.Class}</strong> - {option.Confidence}
                </li>
              ))}
            </ul>
            {data.Warning && (
              <p>
                ⚠️ <strong>Warning:</strong> {data.Warning}
                <br />
                ⚠️{" "}
                <strong>
                  This is not a final prediction, just an approximation.
                </strong>
              </p>
            )}
          </div>
        );

        setMessages((prev) => [
          ...prev,
          { text: formattedPredictions, type: "bot" },
        ]);
      } else if (data.Prediction) {
        const formattedResponse = (
          <div>
            <p>
              🩺 <strong>Diagnosis Result:</strong> {data.Prediction}
            </p>
            <p>
              📊 <strong>Confidence:</strong> {data.Confidence}
            </p>
            <p>
              <strong>
                This is not a final prediction, just an approximation.
              </strong>
            </p>
          </div>
        );

        setMessages((prev) => [
          ...prev,
          { text: formattedResponse, type: "bot" },
        ]);
      } else {
        setError(
          "⚠️ Please upload a proper skin image (try another if possible)."
        );
      }
    } catch (err) {
      console.error("Image Upload Error:", err);
      setError("❌ Failed to analyze image. Try again.");
    } finally {
      setLoading(false);
      setUploadedImage(null);
    }
  };

  const handleAudioRecord = async () => {
    setIsRecording(!isRecording);
    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:5001/speech", {
        method: "POST",
      });
      const data = await response.json();
      setInput(data.transcription);
    } catch (err) {
      console.error("Speech Error:", err);
      setError("❌ Speech recognition failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-section">
        <div className="messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.type}`}>
              {msg.text && <div className="formatted-text">{msg.text}</div>}
              {msg.image && <img src={msg.image} alt="Uploaded" />}
            </div>
          ))}
          {loading && <p className="loading-spinner">⏳ Processing...</p>}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <div className="error-banner">
            {error}
            <button onClick={() => setError("")}>✖️</button>
          </div>
        )}

        <div className="input-section">
          <input
            type="text"
            placeholder="Ask anything..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <label htmlFor="image-upload" className="upload-btn">
            📷
          </label>
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={{ display: "none" }}
          />
          <button
            onClick={handleAudioRecord}
            className={`mic-btn ${isRecording ? "recording" : ""}`}
            disabled={loading}
          >
            <FaMicrophone />
          </button>
          <button onClick={handleSend} disabled={loading}>
            <FaPaperPlane />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
