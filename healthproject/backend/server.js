
import express from "express";
import mongoose from "mongoose";
import dotenv from "dotenv";
import cors from "cors";
import authRoutes from "./routes/auth.js"; // Import authentication routes

dotenv.config();
const app = express();

// Middleware
app.use(express.json()); // To parse JSON request bodies
app.use(cors()); // Enable CORS for frontend connection

// Routes
app.use("/api/auth", authRoutes); // All auth-related routes

// // ✅ Debugging: Print Registered Routes (Including Nested Routes)
// console.log("🔍 Checking Registered Routes...");
// app._router.stack.forEach((layer) => {
//   if (layer.route) {
//     console.log(`✅ ${Object.keys(layer.route.methods).join(", ").toUpperCase()} ${layer.route.path}`);
//   } else if (layer.name === "router" && layer.handle.stack) {
//     layer.handle.stack.forEach((subLayer) => {
//       if (subLayer.route) {
//         console.log(`✅ ${Object.keys(subLayer.route.methods).join(", ").toUpperCase()} ${subLayer.route.path}`);
//       }
//     });
//   }
// });
// console.log("✅ Route Debugging Complete");

// MongoDB Connection & Server Start
const PORT = process.env.PORT || 5000;
mongoose
  .connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => {
    console.log("✅ MongoDB Connected");
    app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`)); // Start server only after DB connection
  })
  .catch((err) => console.error("❌ MongoDB connection error:", err));
