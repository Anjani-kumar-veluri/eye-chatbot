import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  username: String,
  email: { type: String, required: true, unique: true },
  password: String,
  isVerified: Boolean,
  verificationToken: String,
  resetToken: { type: String, default: null },  // ✅ Ensure this exists
  resetTokenExpiry: { type: Date, default: null } // ✅ Ensure this exists
});

export default mongoose.model("User", userSchema);
