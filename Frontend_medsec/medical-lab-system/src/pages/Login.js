import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Button from "../components/Button";
import Spinner from "../components/Spinner";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formError, setFormError] = useState("");
  const { login, loading, error, isAuthenticated, user } = useAuth();
  const navigate = useNavigate();

  // Add responsive state
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;

  // Helper for role-based navigation
  const getRedirectPath = (role) => {
    // Normalize role to lowercase and log it for debugging
    const normalizedRole = (role || "").toLowerCase().trim();
    console.log(
      `Determining redirect for role: ${role} (normalized: ${normalizedRole})`
    );

    // Check specifically for "lab" role
    if (normalizedRole === "lab") {
      console.log("→ Redirecting to lab page: /lab/add-patient");
      return "/lab/add-patient";
    } else {
      console.log("→ Redirecting to doctor page: /doctor/search");
      return "/doctor/search";
    }
  };

  // Handle window resize for responsiveness
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Redirect if already logged in
  useEffect(() => {
    console.log("Login effect - Auth status:", isAuthenticated, "User:", user);

    if (isAuthenticated && user) {
      console.log("User authenticated with role:", user.role);
      const redirectPath = getRedirectPath(user.role);
      console.log("Navigating to:", redirectPath);
      navigate(redirectPath);
    }
  }, [isAuthenticated, navigate, user]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormError("");

    console.log("Form submitted with email:", email);

    if (!email.trim()) {
      setFormError("Email is required");
      return;
    }

    if (!password.trim()) {
      setFormError("Password is required");
      return;
    }

    try {
      console.log("Calling login function");
      const result = await login(email, password);
      console.log("Login successful, full result:", result);

      // The redirect will be handled by the useEffect above when isAuthenticated becomes true
      // but we can also navigate directly as a backup
      const redirectPath = getRedirectPath(result.role);
      console.log("Backup navigation to:", redirectPath);
      navigate(redirectPath);
    } catch (err) {
      console.error("Login error:", err);
      setFormError(err.message || "Login failed");
    }
  };

  // ===== STYLES =====

  // Container styles
  const pageContainerStyle = {
    minHeight: "100vh",
    width: "100%",
    maxWidth: "100%",
    overflowX: "hidden",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "linear-gradient(to bottom, #EFF6FF, #FFFFFF)",
    padding: isMobile ? "16px" : isTablet ? "24px" : "48px",
    position: "relative",
    boxSizing: "border-box",
  };

  // Top header bar style
  const headerBarStyle = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: isMobile ? "24px" : "32px",
    backgroundColor: "#2563EB",
    zIndex: 0,
  };

  // Card container style
  const cardContainerStyle = {
    maxWidth: isMobile ? "100%" : isTablet ? "450px" : "480px",
    width: "100%",
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    boxShadow:
      "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
    padding: isMobile ? "20px 16px" : isTablet ? "28px 24px" : "32px",
    zIndex: 10,
    position: "relative",
    boxSizing: "border-box",
  };

  // Title container style
  const titleContainerStyle = {
    textAlign: "center",
    marginBottom: isMobile ? "24px" : "32px",
  };

  // Title style
  const titleStyle = {
    fontSize: isMobile ? "22px" : isTablet ? "26px" : "30px",
    fontWeight: 800,
    color: "#2563EB",
    marginBottom: "8px",
  };

  // Title divider style
  const titleDividerStyle = {
    height: "3px",
    width: isMobile ? "50px" : "60px",
    backgroundColor: "#2563EB",
    margin: "0 auto 16px auto",
  };

  // Subtitle style
  const subtitleStyle = {
    color: "#4B5563",
    fontSize: isMobile ? "13px" : isTablet ? "14px" : "16px",
    padding: "0 8px",
  };

  // Form style
  const formStyle = {
    marginTop: isMobile ? "16px" : "24px",
    width: "100%",
  };

  // Form field group style
  const formGroupStyle = {
    marginBottom: isMobile ? "16px" : "20px",
    width: "100%",
  };

  // Field label style
  const labelStyle = {
    display: "block",
    marginBottom: "6px",
    fontWeight: 500,
    fontSize: isMobile ? "14px" : "15px",
    color: "#374151",
  };

  // Input style
  const inputStyle = {
    width: "100%",
    padding: isMobile ? "10px 12px" : "12px 16px",
    fontSize: isMobile ? "14px" : "15px",
    border: "1px solid #D1D5DB",
    borderRadius: "8px",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.2s ease, box-shadow 0.2s ease",
  };

  // Input focus style - applies on focus
  const focusStyle = (e) => {
    e.target.style.borderColor = "#3B82F6";
    e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.1)";
  };

  // Input blur style - removes focus styles
  const blurStyle = (e) => {
    e.target.style.borderColor = "#D1D5DB";
    e.target.style.boxShadow = "none";
  };

  // Error message container
  const errorContainerStyle = {
    backgroundColor: "#FEF2F2",
    border: "1px solid #FECACA",
    borderRadius: "8px",
    padding: isMobile ? "10px 12px" : "12px 16px",
    marginBottom: isMobile ? "16px" : "20px",
    color: "#DC2626",
    fontSize: isMobile ? "13px" : "14px",
    display: "flex",
    alignItems: "center",
    boxSizing: "border-box",
    width: "100%",
  };

  // Error icon style
  const errorIconStyle = {
    width: isMobile ? "16px" : "20px",
    height: isMobile ? "16px" : "20px",
    marginRight: "8px",
    flexShrink: 0,
  };

  // Button wrapper style
  const buttonWrapperStyle = {
    marginTop: isMobile ? "20px" : "24px",
    width: "100%",
  };

  // Footer style
  const footerStyle = {
    marginTop: isMobile ? "24px" : isTablet ? "28px" : "32px",
    textAlign: "center",
    fontSize: isMobile ? "12px" : "14px",
    color: "#6B7280",
    width: "100%",
    maxWidth: "100%",
    padding: "0 16px",
    boxSizing: "border-box",
  };

  return (
    <div style={pageContainerStyle}>
      <div style={headerBarStyle}></div>

      <div style={cardContainerStyle}>
        <div style={titleContainerStyle}>
          <h1 style={titleStyle}>MedSec System</h1>
          <div style={titleDividerStyle}></div>
          <p style={subtitleStyle}>Sign in to access the medSec system</p>
        </div>

        <form onSubmit={handleSubmit} style={formStyle}>
          {/* Display error if any */}
          {(error || formError) && (
            <div style={errorContainerStyle}>
              <svg
                style={errorIconStyle}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{formError || error}</span>
            </div>
          )}

          <div style={formGroupStyle}>
            <label style={labelStyle} htmlFor="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              style={inputStyle}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              onFocus={focusStyle}
              onBlur={blurStyle}
            />
          </div>

          <div style={formGroupStyle}>
            <label style={labelStyle} htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              style={inputStyle}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              onFocus={focusStyle}
              onBlur={blurStyle}
            />
          </div>

          <div style={buttonWrapperStyle}>
            <Button
              type="submit"
              variant="primary"
              disabled={loading}
              style={{
                width: "100%",
                padding: isMobile ? "12px" : "14px",
                fontSize: isMobile ? "15px" : "16px",
                borderRadius: "8px",
              }}
            >
              {loading ? (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Spinner size="small" color="white" />
                  <span style={{ marginLeft: "8px" }}>Signing in...</span>
                </div>
              ) : (
                "Sign In"
              )}
            </Button>
          </div>
        </form>

        <div style={footerStyle}>
          <p>© 2025 MedSec System | All Rights Reserved</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
