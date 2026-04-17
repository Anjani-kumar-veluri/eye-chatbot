import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";

function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [message, setMessage] = useState("Verifying email...");

  useEffect(() => {
    const verifyUser = async () => {
      const token = searchParams.get("token"); // Get token from URL
      if (!token) {
        setMessage("Invalid or missing token.");
        return;
      }

      try {
        const response = await axios.post("http://localhost:5000/api/auth/verify-email", { token });
        if (response.data.success) {
          setMessage("Email verified successfully! Redirecting...");
          setTimeout(() => navigate("/"), 3000); // Redirect to login after 3 sec
        } else {
          setMessage("Verification failed. Please try again.");
        }
      } catch (error) {
        setMessage("Verification failed. Token may be expired.");
      }
    };

    verifyUser();
  }, [searchParams, navigate]);

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>{message}</h2>
    </div>
  );
}

export default VerifyEmail;
