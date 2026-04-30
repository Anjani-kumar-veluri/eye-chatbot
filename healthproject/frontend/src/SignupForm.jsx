import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './SignupForm.css';

function SignupForm() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // Regex for strong password
  // const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');

    // Validate password strength
    // if (!passwordRegex.test(formData.password)) {
    //   setError(
    //     'Password must be at least 8 characters long, with one uppercase, one lowercase, one number, and one special character.'
    //   );
    //   console.log("PASSWORD ERROR");
      
    //   return;
    // }

    try {
      const response = await axios.post('http://localhost:5000/api/auth/signup', formData);
      if (response.data.success) {
        setSuccessMessage('Signup successful! Check your email to verify your account.');
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Signup failed. Try again.');
    }


  };

  return (
    <center>
      <div className="container-wrapper1">
        <div className="container1">
          <div className="signup-section">
            <h2>SIGN UP</h2>
            {error && <p className="error">{error}</p>}
            {successMessage && <p className="success">{successMessage}</p>}
            <form onSubmit={handleSubmit}>
              <div className="input-container">
                <input 
                  type="text" 
                  name="username"
                  placeholder="User name" 
                  value={formData.username} 
                  onChange={handleChange} 
                  required 
                />
                <span className="icon">&#128100;</span>
              </div>
              <div className="input-container">
                <input 
                  type="email" 
                  name="email"
                  placeholder="Email" 
                  value={formData.email} 
                  onChange={handleChange} 
                  required 
                />
                <span className="icon">&#9993;</span>
              </div>
              <div className="input-container">
                <input 
                  type="password" 
                  name="password"
                  placeholder="Password" 
                  value={formData.password} 
                  onChange={handleChange} 
                  required 
                />
                <span className="icon">&#128274;</span>
              </div>
              <button type="submit" className="signup_submit">SIGN UP</button>
            </form>
            <br />
            <div className="options">
              <Link to="/">Already have an Account? Login.</Link>
            </div>
          </div>
          <div className="image-section">
            <img src="robo3.png" alt="Futuristic Robot" />
          </div>
        </div>
      </div>
    </center>
  );
}

export default SignupForm;
