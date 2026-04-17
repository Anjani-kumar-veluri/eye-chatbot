import React from 'react';
import './Image_3d.css';


const ImagePage = () => {
  return (
    <div className="container-wrapper5">
    <div className="container5">
      <div className="image-section">
        <img src="robot.png" alt="3D Overlap" className="overlap-image" />
      </div>
      <div className="login-section">
        <h2>Welcome Back!</h2>
        <form>
          <input type="text" placeholder="Username" />
          <input type="password" placeholder="Password" />
          <button type="submit">Login</button>
        </form>
      </div>
    </div>
    </div>
  );
};

export default ImagePage;
