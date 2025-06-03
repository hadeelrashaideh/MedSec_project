import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const Header = () => {
  const { user, logout, isLab, isDoctor } = useAuth();
  const navigate = useNavigate();

  // Add responsive state
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;

  // Handle window resize for responsiveness
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      if (window.innerWidth >= 768) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleLogout = () => {
    logout();
    navigate("/login");
    setMobileMenuOpen(false);
  };

  // ===== STYLES =====

  // Header container styles
  const headerContainerStyle = {
    backgroundColor: "#FFFFFF",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.05)",
    position: "sticky",
    top: 0,
    width: "100%",
    zIndex: 100,
  };

  // Inner content wrapper
  const headerInnerStyle = {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: isMobile ? "0 16px" : isTablet ? "0 24px" : "0 32px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    height: isMobile ? "60px" : "70px",
  };

  // Logo container
  const logoContainerStyle = {
    display: "flex",
    alignItems: "center",
  };

  // Logo style
  const logoStyle = {
    fontSize: isMobile ? "18px" : "22px",
    fontWeight: "700",
    color: "#2563EB",
    textDecoration: "none",
    display: "flex",
    alignItems: "center",
  };

  // Logo icon style
  const logoIconStyle = {
    width: isMobile ? "18px" : "24px",
    height: isMobile ? "18px" : "24px",
    marginRight: "8px",
    color: "#2563EB",
    flexShrink: 0,
  };

  // Navigation container
  const navContainerStyle = {
    display: "flex",
    alignItems: "center",
  };

  // Welcome message style
  const welcomeMessageStyle = {
    marginRight: "16px",
    fontSize: "14px",
    fontWeight: "400",
    color: "#64748B",
    display: isMobile ? "none" : "block",
  };

  // Username style
  const usernameStyle = {
    fontWeight: "600",
    color: "#334155",
  };

  // Role badge style
  const roleBadgeStyle = {
    marginLeft: "8px",
    padding: "3px 8px",
    fontSize: "11px",
    fontWeight: "500",
    backgroundColor: "#DBEAFE",
    color: "#1E40AF",
    borderRadius: "12px",
    display: "inline-block",
    textTransform: "uppercase",
  };

  // Nav links container
  const navLinksStyle = {
    display: isMobile ? (mobileMenuOpen ? "flex" : "none") : "flex",
    flexDirection: isMobile ? "column" : "row",
    alignItems: isMobile ? "flex-start" : "center",
    position: isMobile ? "absolute" : "static",
    top: isMobile ? "60px" : "auto",
    left: 0,
    right: 0,
    backgroundColor: "#FFFFFF",
    padding: isMobile ? "16px" : 0,
    boxShadow: isMobile ? "0 4px 6px -1px rgba(0, 0, 0, 0.1)" : "none",
    zIndex: 99,
  };

  // Nav link base style
  const getNavLinkStyle = (isLogout = false) => ({
    padding: isMobile ? "12px 16px" : "8px 12px",
    borderRadius: "6px",
    fontSize: "14px",
    fontWeight: "500",
    textDecoration: "none",
    marginLeft: isMobile ? 0 : "8px",
    marginBottom: isMobile ? "8px" : 0,
    display: "block",
    width: isMobile ? "100%" : "auto",
    textAlign: isMobile ? "left" : "center",
    backgroundColor: isLogout
      ? "rgba(239, 68, 68, 0.1)"
      : "rgba(59, 130, 246, 0.1)",
    color: isLogout ? "#DC2626" : "#2563EB",
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s ease",
  });

  // Hover effects for links (applied with inline functions)
  const linkHoverStyle = (e, isLogout = false) => {
    e.currentTarget.style.backgroundColor = isLogout
      ? "rgba(239, 68, 68, 0.2)"
      : "rgba(59, 130, 246, 0.2)";
  };

  const linkLeaveStyle = (e, isLogout = false) => {
    e.currentTarget.style.backgroundColor = isLogout
      ? "rgba(239, 68, 68, 0.1)"
      : "rgba(59, 130, 246, 0.1)";
  };

  // Login button style
  const loginButtonStyle = {
    padding: "8px 16px",
    borderRadius: "6px",
    fontSize: "14px",
    fontWeight: "600",
    backgroundColor: "#2563EB",
    color: "#FFFFFF",
    border: "none",
    textDecoration: "none",
    cursor: "pointer",
    transition: "all 0.2s ease",
  };

  // Mobile menu toggle button style
  const menuToggleStyle = {
    display: isMobile ? "block" : "none",
    background: "transparent",
    border: "none",
    padding: "8px",
    cursor: "pointer",
  };

  // Mobile menu icon style
  const menuIconStyle = {
    width: "24px",
    height: "24px",
    color: "#64748B",
  };

  // Get dashboard link based on user role
  const getDashboardLink = () => {
    if (isLab) {
      return "/lab/add-patient";
    } else if (isDoctor) {
      return "/doctor/search";
    }
    return "/";
  };

  return (
    <header style={headerContainerStyle}>
      <div style={headerInnerStyle}>
        <div style={logoContainerStyle}>
          <Link to={user ? getDashboardLink() : "/"} style={logoStyle}>
            <svg
              style={logoIconStyle}
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M19 6h-4a1 1 0 100 2h4a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2zm9 5h-5a1 1 0 100 2h5a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2zm9 5h-4a1 1 0 100 2h4a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2z" />
              <path d="M11.3 3.3a1 1 0 00-1.4 1.4l1.3 1.3H3a1 1 0 000 2h8.2l-1.3 1.3a1 1 0 001.4 1.4l3-3a1 1 0 000-1.4l-3-3zM19 16h-8.2l1.3-1.3a1 1 0 10-1.4-1.4l-3 3a1 1 0 000 1.4l3 3a1 1 0 001.4-1.4l-1.3-1.3H19a1 1 0 100-2z" />
            </svg>
            MedSec
          </Link>
        </div>

        <div style={navContainerStyle}>
          {isMobile && (
            <button
              style={menuToggleStyle}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <svg
                  style={menuIconStyle}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg
                  style={menuIconStyle}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16m-7 6h7"
                  />
                </svg>
              )}
            </button>
          )}

          <div style={navLinksStyle}>
            {user ? (
              <>
                <span style={welcomeMessageStyle}>
                  Welcome, <span style={usernameStyle}>{user.email}</span>
                  <span style={roleBadgeStyle}>{user.role}</span>
                </span>

                <button
                  onClick={handleLogout}
                  style={getNavLinkStyle(true)}
                  onMouseOver={(e) => linkHoverStyle(e, true)}
                  onMouseOut={(e) => linkLeaveStyle(e, true)}
                >
                  <div style={{ display: "flex", alignItems: "center" }}>
                    <svg
                      style={{
                        width: "16px",
                        height: "16px",
                        marginRight: "6px",
                      }}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                      />
                    </svg>
                    Logout
                  </div>
                </button>
              </>
            ) : (
              <Link
                to="/login"
                style={loginButtonStyle}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = "#1D4ED8";
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = "#2563EB";
                }}
              >
                <div style={{ display: "flex", alignItems: "center" }}>
                  <svg
                    style={{
                      width: "16px",
                      height: "16px",
                      marginRight: "6px",
                    }}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
                    />
                  </svg>
                  Login
                </div>
              </Link>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
