import React, { useState } from "react";

const Input = ({
  label,
  id,
  type = "text",
  value,
  onChange,
  placeholder = "",
  required = false,
  error = null,
  className = "",
  fullWidth = true,
  disabled = false,
}) => {
  // Add state for focus
  const [isFocused, setIsFocused] = useState(false);

  // Base styles for the input container
  const containerStyle = {
    marginBottom: "16px",
    width: fullWidth ? "100%" : "auto",
  };

  // Label styles
  const labelStyle = {
    display: "block",
    marginBottom: "4px",
    fontWeight: 500,
    fontSize: "14px",
    color: "#374151", // gray-700
  };

  // Get border color based on state
  const getBorderColor = () => {
    if (error) return "#ef4444"; // red-500
    if (isFocused) return "#3b82f6"; // blue-500
    return "#d1d5db"; // gray-300
  };

  // Input styles
  const inputStyle = {
    width: fullWidth ? "100%" : "auto",
    padding: "8px 16px",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: getBorderColor(),
    borderRadius: "6px",
    fontSize: "16px",
    lineHeight: "1.5",
    transition: "all 0.2s ease",
    backgroundColor: disabled ? "#f3f4f6" : "#ffffff", // gray-100 : white
    color: disabled ? "#6b7280" : "#111827", // gray-500 : gray-900
    outline: "none",
    boxShadow: isFocused ? "0 0 0 3px rgba(59, 130, 246, 0.3)" : "none", // blue-500 with opacity when focused
  };

  // Error message styles
  const errorStyle = {
    marginTop: "4px",
    fontSize: "14px",
    color: "#ef4444", // red-500
  };

  // Required asterisk styles
  const requiredStyle = {
    color: "#ef4444", // red-500
    marginLeft: "2px",
  };

  return (
    <div style={containerStyle}>
      {label && (
        <label style={labelStyle} htmlFor={id}>
          {label} {required && <span style={requiredStyle}>*</span>}
        </label>
      )}

      <input
        id={id}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
        style={inputStyle}
        className={className}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      />

      {error && <p style={errorStyle}>{error}</p>}
    </div>
  );
};

export default Input;
