import React, { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";

const EncryptionStatus = ({ showLabel = true, size = "medium" }) => {
  const { encryptionStatus } = useAuth();

  // Determine classes based on size prop
  const getSizeClasses = () => {
    switch (size) {
      case "small":
        return {
          container: { padding: "4px 6px", fontSize: "10px" },
          icon: { width: "10px", height: "10px", marginRight: "4px" },
        };
      case "large":
        return {
          container: { padding: "6px 12px", fontSize: "14px" },
          icon: { width: "16px", height: "16px", marginRight: "8px" },
        };
      case "medium":
      default:
        return {
          container: { padding: "4px 8px", fontSize: "12px" },
          icon: { width: "14px", height: "14px", marginRight: "6px" },
        };
    }
  };

  const sizeClasses = getSizeClasses();

  const containerStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "500",
    color: encryptionStatus ? "#047857" : "#6B7280",
    backgroundColor: encryptionStatus ? "#D1FAE5" : "#F3F4F6",
    borderRadius: "4px",
    ...sizeClasses.container,
  };

  const iconStyle = {
    ...sizeClasses.icon,
  };

  return (
    <div
      style={containerStyle}
      title={
        encryptionStatus
          ? "End-to-End Encryption is active"
          : "Standard Encryption"
      }
    >
      <svg
        style={iconStyle}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
        />
      </svg>
      {showLabel && (
        <span>
          {encryptionStatus ? "End-to-End Encryption" : "Standard Encryption"}
        </span>
      )}
    </div>
  );
};

export default EncryptionStatus;
