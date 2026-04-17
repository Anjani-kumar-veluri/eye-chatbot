import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './LoginForm.css';

const ResetPassword = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const token = queryParams.get('token');
  const email = queryParams.get('email');

  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const handleResetPassword = async (e) => {
    e.preventDefault();

    if (!newPassword || !confirmPassword) {
      setError('Please enter both password fields.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, email, newPassword }),
      });

      const data = await response.json();
      console.log("🔹 Server Response:", data);  // ✅ Log API response

      if (response.ok) {
        setMessage('Password reset successful! Redirecting to login...');
        setError('');

        setTimeout(() => {
          console.log("✅ Redirecting to /");
          navigate('/');  // ✅ Navigating to the login page
        }, 3000);
      } else {
        console.error("❌ Reset Password Error:", data.message);
        setError(data.message);
      }
    } catch (error) {
      console.error("❌ Reset Password Fetch Error:", error);
      setError('Error resetting password. Please try again.');
    }
  };

  return (
    <div className="container-wrapper">
      <div className="container2">
        <div className="image-section1">
          <img src="robo2.png" alt="Futuristic Robot" />
        </div>
        <div className="login-section1">
          <h2>Reset Password</h2>
          <form onSubmit={handleResetPassword}>
            <div className="input-container">
              <input type="password" placeholder="New Password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
              <span className="icon">&#128274;</span>
            </div>
            <div className="input-container">
              <input type="password" placeholder="Confirm Password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} />
              <span className="icon">&#128274;</span>
            </div>
            {error && <p className="error-message">{error}</p>}
            {message && <p className="success-message">{message}</p>}
            <button type="submit" className="login_submit">Reset Password</button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ResetPassword;
