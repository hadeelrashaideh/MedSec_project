import React, { useState, useEffect } from "react";
import Header from "./Header";

const Layout = ({ children }) => {
  // Add responsive state
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;

  // Handle window resize for responsiveness
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // ==== STYLES ====

  // Main layout container
  const layoutContainerStyle = {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#F8FAFC",
    width: "100%",
    maxWidth: "100%",
    overflow: "hidden",
  };

  // Main content area
  const mainContentStyle = {
    flex: "1",
    width: "100%",
    padding: isMobile ? "16px 0" : isTablet ? "24px 0" : "32px 0",
  };

  // Footer container
  const footerContainerStyle = {
    backgroundColor: "#FFFFFF",
    borderTop: "1px solid #E2E8F0",
    padding: isMobile ? "16px 0" : "24px 0",
    width: "100%",
    zIndex: 10,
    boxShadow: "0 -2px 10px rgba(0, 0, 0, 0.03)",
    marginTop: "auto",
  };

  // Footer inner wrapper
  const footerInnerStyle = {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "0 16px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    textAlign: "center",
  };

  // Footer logo and brand
  const footerBrandStyle = {
    marginBottom: "8px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  };

  // Footer logo
  const footerLogoStyle = {
    color: "#2563EB",
    fontWeight: "700",
    fontSize: "18px",
    display: "flex",
    alignItems: "center",
    marginBottom: "4px",
  };

  // Footer logo icon
  const footerLogoIconStyle = {
    width: "16px",
    height: "16px",
    marginRight: "6px",
    color: "#2563EB",
  };

  // Footer subtitle
  const footerSubtitleStyle = {
    color: "#64748B",
    fontSize: "14px",
  };

  // Footer copyright
  const footerCopyrightStyle = {
    color: "#64748B",
    fontSize: "13px",
    marginTop: "8px",
  };

  return (
    <div style={layoutContainerStyle}>
      <Header />
      <main style={mainContentStyle}>{children}</main>
      <footer style={footerContainerStyle}>
        <div style={footerInnerStyle}>
          <div style={footerBrandStyle}>
            <div style={footerLogoStyle}>
              <svg
                style={footerLogoIconStyle}
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M19 6h-4a1 1 0 100 2h4a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2zm9 5h-5a1 1 0 100 2h5a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2zm9 5h-4a1 1 0 100 2h4a1 1 0 100-2zm-9 0H5a1 1 0 000 2h5a1 1 0 100-2z" />
              </svg>
              MedSec
            </div>
            <div style={footerSubtitleStyle}>Medical Laboratory System</div>
          </div>
          <div style={footerCopyrightStyle}>
            &copy; {new Date().getFullYear()} MedSec System. All rights
            reserved.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
