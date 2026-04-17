import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LoginForm from './LoginForm';
import SignupForm from './SignupForm';
import WelcomePage from './WelcomePage';
import InfoPage from './InfoPage';
import ChatPage from './ChatbotPage';
import VerifyEmail from "./VerifyEmail";
import ResetPassword from './ResetPassword';


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginForm />} />
        <Route path="/signup" element={<SignupForm />} />
        <Route path="/welcome" element={<WelcomePage/>}/>
        <Route path="/info" element={<InfoPage/>}/>
        <Route path="/chat" element={<ChatPage/>}/>
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="/reset-password" element={<ResetPassword />} />
      </Routes>
    </Router>
  );
}

export default App;
