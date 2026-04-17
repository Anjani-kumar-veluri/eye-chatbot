import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import User from '../models/user.js';
import dotenv from 'dotenv';
import nodemailer from 'nodemailer';

dotenv.config();
const router = express.Router();

// 📌 Configure Nodemailer
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER, 
    pass: process.env.EMAIL_PASS, 
  },
});

// 📌 Signup Route with Email Verification
router.post('/signup', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    console.log("Received signup request:", { username, email });

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      console.log("User already exists:", email);
      return res.status(400).json({ message: 'User already exists' });
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 12);
    console.log("Password hashed successfully");

    // Generate email verification token
    const verificationToken = jwt.sign({ email }, process.env.JWT_SECRET, { expiresIn: '7d' });
    console.log("Generated verification token:", verificationToken);

    // Create new user (unverified by default)
    const newUser = new User({
      username,
      email,
      password: hashedPassword,
      isVerified: false, // User is not verified initially
      verificationToken, // Store token in DB
    });

    await newUser.save();
    console.log("User saved to database:", email);

    // ✅ Corrected verification link
    const verificationLink = `http://localhost:5000/api/auth/verify-email?token=${verificationToken}`;
    console.log("Verification link:", verificationLink);

    // Send verification email
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Verify Your Email',
      html: `<p>Click the link below to verify your email:</p>
             <a href="${verificationLink}" target="_blank">${verificationLink}</a>`,
    };

    transporter.sendMail(mailOptions, (error, info) => {
      if (error) {
        console.error("Error sending email:", error);
        return res.status(500).json({ message: 'Error sending verification email' });
      } else {
        console.log("Verification email sent:", info.response);
        res.status(201).json({ message: 'User created successfully. Verification email sent.' });
      }
    });

  } catch (error) {
    console.error("Signup Error:", error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// 📌 Email Verification Route (Fixed)
router.get('/verify-email', async (req, res) => {
    try {
      const { token } = req.query;
      console.log("Received verification request for token:", token);
  
      if (!token) {
        console.log("No token received.");
        return res.status(400).json({ message: "No token provided." });
      }
  
      let decoded;
      try {
        decoded = jwt.verify(token, process.env.JWT_SECRET);
        console.log("Decoded token:", decoded);
      } catch (error) {
        console.error("JWT Verification Error:", error);
        return res.status(400).json({ message: "Verification failed. Token may be expired or invalid." });
      }
  
      const user = await User.findOne({ email: decoded.email, verificationToken: token });
      if (!user) {
        console.log("No user found with this email or token mismatch.");
        return res.status(400).json({ message: "Invalid or expired token" });
      }
  
      user.isVerified = true;
      user.verificationToken = null;
      await user.save();
      console.log("User verified successfully:", user.email);
  
      res.redirect("http://localhost:5173/");
    } catch (error) {
      console.error("Verification Error:", error);
      res.status(400).json({ message: "Invalid or expired token" });
    }
});

// 📌 Login Route (Only for Verified Users)
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    console.log("Login request received for:", email);

    // Check if user exists
    const user = await User.findOne({ email });
    if (!user) {
      console.log("User not found:", email);
      return res.status(404).json({ message: 'User not found' });
    }

    // 📌 Check if user is verified
    if (!user.isVerified) {
      console.log("User not verified:", email);
      return res.status(403).json({ message: 'Please verify your email before logging in.' });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      console.log("Invalid password attempt for:", email);
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '1h' } // Token expires in 1 hour
    );

    console.log("Login successful for:", email);
    res.status(200).json({ message: 'Login successful', token });
  } catch (error) {
    console.error("Login Error:", error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});
import crypto from 'crypto';

router.post('/forgot-password', async (req, res) => {
  try {
    const { email } = req.body;
    console.log("🔹 Forgot password request received for:", email);

    // Check if user exists
    const user = await User.findOne({ email });
    if (!user) {
      console.log("❌ User not found:", email);
      return res.status(404).json({ message: 'User not found' });
    }

    // Generate Reset Token
    const resetToken = crypto.randomBytes(32).toString('hex');
    user.resetToken = resetToken;
    user.resetTokenExpiry = Date.now() + 3600000; // 1-hour expiry

    // ✅ Save the user with the reset token in MongoDB
    await user.save();  
    console.log("✅ Reset token stored in MongoDB for:", email);

    // Create Reset Link
    const resetLink = `http://localhost:5173/reset-password?token=${resetToken}&email=${email}`;
    console.log("🔗 Password reset link:", resetLink);

    // Send Email
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Reset Your Password',
      html: `<p>Click the link below to reset your password:</p>
             <a href="${resetLink}" target="_blank">${resetLink}</a>`,
    };

    await transporter.sendMail(mailOptions);
    console.log("✅ Password reset email sent successfully");

    return res.status(200).json({ message: 'Password reset email sent successfully.' });

  } catch (error) {
    console.error("❌ Forgot Password Error:", error);
    return res.status(500).json({ message: 'Server error', error: error.message });
  }
});
router.post('/reset-password', async (req, res) => {
  try {
      const { token, email, newPassword } = req.body;
      console.log("🔹 Reset Password Request received for:", email);

      const user = await User.findOne({ email, resetToken: token });

      if (!user || !user.resetTokenExpiry || user.resetTokenExpiry < Date.now()) {
          console.log("❌ Invalid or expired reset token.");
          return res.status(400).json({ message: "Invalid or expired reset token" });
      }

      // Hash new password
      const hashedPassword = await bcrypt.hash(newPassword, 12);
      user.password = hashedPassword;
      user.resetToken = null;  
      user.resetTokenExpiry = null;
      await user.save();

      console.log("✅ Password reset successfully for:", email);

      // ✅ Send a response with a redirect URL
      res.status(200).json({
          message: "Password reset successful!",
          redirectUrl: "/"
      });

  } catch (error) {
      console.error("❌ Reset Password Error:", error);
      res.status(500).json({ message: "Server error", error: error.message });
  }
});


export default router;
