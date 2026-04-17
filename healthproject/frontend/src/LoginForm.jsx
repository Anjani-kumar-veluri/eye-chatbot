import React, { useState } from 'react';
import './LoginForm.css';
import { useNavigate } from 'react-router-dom';
import { Link } from 'react-router-dom';

const LoginForm = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState(''); // Success message for forgot password

  // Handle login
  const handleLogin = async (e) => {
    e.preventDefault();

    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();

      if (data.token) {
        localStorage.setItem('token', data.token);
        navigate('/welcome');
      } else {
        setError(data.message);
      }
    } catch (error) {
      console.error('Login Error:', error);
      setError('An error occurred during login. Please try again.');
    }
  };

  // Handle forgot password request
  const handleForgotPassword = async (e) => {
    e.preventDefault(); // Prevent default link behavior

    if (!email) {
      setError('Please enter your email to reset password.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('Password reset link sent! Check your email.');
        setError('');
      } else {
        setError(data.message);
      }
    } catch (error) {
      console.error('Forgot Password Error:', error);
      setError('Error sending password reset email.');
    }
  };

  return (
    <div className="container-wrapper">
      <div className="container2">
        <div className="image-section1">
          <img src="robo2.png" alt="Futuristic Robot" />
        </div>
        <div className="login-section1">
          <h2>LOGIN</h2>
          <form onSubmit={handleLogin}>
            <div className="input-container">
              <input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <span className="icon">&#9993;</span>
            </div>
            <div className="input-container">
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <span className="icon">&#128274;</span>
            </div>
            {error && <p className="error-message">{error}</p>}
            {message && <p className="success-message">{message}</p>} {/* Success Message */}
            <button type="submit" className="login_submit">LOGIN</button>
          </form>
          <div className="options1">
            <div><Link to="/signup">Create an Account?</Link></div>
            <div>
              <a href="#" onClick={handleForgotPassword}>Forgot Password?</a> {/* No structure change */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;
