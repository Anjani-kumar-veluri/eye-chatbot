import React from 'react';
import './Info_Page.css';
import image_info from './assets/image_info.png';
import robo_logo from './assets/robo_logo.png';
import {useNavigate} from 'react-router-dom';

const Info = () => {
  const navigate = useNavigate();
      const handleContinue = (e) => {
        e.preventDefault();
        navigate('/chat');
      }

  return (
    <center>
    <div className="info-container">
      <h1 className="text1">Your Health</h1>
      <h2 className="text2">Your AI Companion! <img src={robo_logo} alt="ricon" className="robo_icon"/><icon></icon></h2>
      
      <div className="content-section">
        <img src={image_info} alt="AI Companion" className="ai-image" />
        <div className="text-section">
          <p className="text3">
            <em>
              Using this software you can ask any health-related questions and receive articles using Artificial Intelligence assistant.
            </em>
          </p>
          <div className="btn">
          <button className="continue-btn" onClick={handleContinue} >
            <div className="left-div" >Continue</div> <div className="right-div">→</div> 
          </button>
          </div>
        </div>
      </div>
    </div>
    </center>
  );
};

export default Info;
