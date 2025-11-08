import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Bot, User, Loader, Sparkles, MessageCircle } from 'lucide-react';
import './App.css';

function App() {
    const [userId] = useState(() => {
    return `${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
  });
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hey! I’m your Success Navigator : your AI mentor. I’m here to guide you in choosing the right field for a brighter future. So, let’s start! What’s your name?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat', {
        user_input: inputValue,
        user_id: userId
      });

      // const botMessage = {
      //   id: Date.now() + 1,
      //   text: response.data.message,
      //   sender: 'bot',
      //   timestamp: new Date(),
      //   careers: response.data.career_recommendations?.slice(0, 5) || []
      // };

      const botMessageData = parseBotMessage(response.data.message);

const botMessage = {
  id: Date.now() + 1,
  text: botMessageData.text,   // pura message + explanation
  sender: 'bot',
  timestamp: new Date(),
  careers: botMessageData.careers
};
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I'm having trouble connecting right now. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  function cleanText(text) {
  // **ke andar jo bhi hai usko hata do including **
  return text.replace(/\*\*.*?\*\*/g, "");
}

  const parseBotMessage = (text) => {
  // Extract all **bold** parts as careers
  const careers = (text.match(/\*\*(.*?)\*\*/g) || [])
    .map(c => c.replace(/\*\*/g, '').trim());

  return { text, careers };
};

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="header-content">
            <div className="bot-avatar">
              <Bot size={24} />
            </div>
            <div className="header-text">
              <h1>Success Navigator</h1>
              <p>Your AI Mentor</p>
            </div>
            <div className="header-icon">
              <Sparkles size={20} />
            </div>
          </div>
        </div>

        <div className="messages-container">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender} fade-in`}>
              <div className="message-avatar">
                {message.sender === 'bot' ? <Bot size={20} /> : <User size={20} />}
              </div>
              <div className="message-content">
                <div className="message-text">{cleanText(message.text)}</div>
                {message.careers && message.careers.length > 0 && (
                  <div className="careers-suggestions">
                    <h4>Top Career Recommendations:</h4>
                    <div className="careers-grid">
                      {message.careers.map((career, index) => (
                        <div key={index} className="career-card">
                          <MessageCircle size={16} />
                          <span>{career}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot fade-in">
              <div className="message-avatar">
                <Bot size={20} />
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <Loader className="pulse" size={16} />
                  <span>Loading...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type here..."
              className="message-input"
              rows="1"
             
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="send-button"
            >
              <Send size={10} />
            </button>
          </div>
          <div className="input-footer">
            <p>Press Enter to send, Shift+Enter for new line</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

