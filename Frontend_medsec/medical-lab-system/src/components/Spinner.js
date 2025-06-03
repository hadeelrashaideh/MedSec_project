import React from "react";

const Spinner = ({ size = "medium", color = "blue" }) => {
  const sizeDimensions = {
    small: "1rem",
    medium: "2rem",
    large: "3rem",
  };

  const colorValues = {
    blue: "#2563eb",
    white: "#ffffff",
    gray: "#4b5563",
    green: "#059669",
  };

  const containerStyles = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  };

  const spinnerStyles = {
    width: sizeDimensions[size],
    height: sizeDimensions[size],
    color: colorValues[color],
    animation: "spin 1s linear infinite",
  };

  return (
    <div style={containerStyles}>
      <svg
        style={spinnerStyles}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          style={{ opacity: 0.25 }}
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          style={{ opacity: 0.75 }}
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};

export default Spinner;
