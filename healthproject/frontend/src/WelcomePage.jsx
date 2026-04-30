import React from 'react';
import { useNavigate } from 'react-router-dom';
import './WelcomePage.css';
import image from './assets/image.png';

const WelcomePage = () => {
  const navigate = useNavigate();

  const handleNavigation = () => {
    navigate('/info'); // Navigate to login page
  };

  return (
    <center>
    <div className="cc">
      <div className="container_wel" onClick={handleNavigation}>
        <img src={image} alt="AI Health Care Chatbot" className="robot3" />
        <p className="text">AI Opthamology Chatbot</p>
      </div>
    </div>
    </center>
  );
};

export default WelcomePage;
