import React, { useState } from "react";

// Button color constants
const COLORS = {
  primary: {
    background: "#0284c7",
    hover: "#0369a1",
    active: "#075985",
    text: "#ffffff",
  },
  secondary: {
    background: "#ede9fe",
    hover: "#ddd6fe",
    active: "#c4b5fd",
    text: "#6d28d9",
  },
  danger: {
    background: "#ef4444",
    hover: "#b91c1c",
    active: "#991b1b",
    text: "#ffffff",
  },
  success: {
    background: "#10b981",
    hover: "#047857",
    active: "#047557",
    text: "#ffffff",
  },
};

// Button size constants
const SIZES = {
  small: {
    height: "32px",
    padding: "0 12px",
    fontSize: "14px",
    minWidth: "70px",
  },
  medium: {
    height: "40px",
    padding: "0 16px",
    fontSize: "16px",
    minWidth: "90px",
  },
  large: {
    height: "48px",
    padding: "0 24px",
    fontSize: "18px",
    minWidth: "110px",
  },
};

const Button = ({
  children,
  variant = "primary",
  size = "medium",
  onClick,
  fullWidth = false,
  disabled = false,
  type = "button",
  icon = null,
  iconPosition = "left",
  className = "",
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isActive, setIsActive] = useState(false);

  // Get color and size settings
  const colors = COLORS[variant] || COLORS.primary;
  const sizeObj = SIZES[size] || SIZES.medium;

  // Determine current background color based on state
  const getBgColor = () => {
    if (disabled) return colors.background;
    if (isActive) return colors.active;
    if (isHovered) return colors.hover;
    return colors.background;
  };

  // Base styles
  const baseStyle = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: getBgColor(),
    color: colors.text,
    borderRadius: "6px",
    fontWeight: 500,
    transition: "all 0.2s ease",
    border: "none",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.6 : 1,
    height: sizeObj.height,
    padding: sizeObj.padding,
    fontSize: sizeObj.fontSize,
    minWidth: sizeObj.minWidth,
    width: fullWidth ? "100%" : "auto",
    position: "relative",
    overflow: "hidden",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
    transform:
      isHovered && !isActive && !disabled
        ? "translateY(-1px)"
        : "translateY(0)",
  };

  // Styles for icon positioning
  const iconStyle = {
    marginRight: iconPosition === "left" && children ? "8px" : 0,
    marginLeft: iconPosition === "right" && children ? "8px" : 0,
  };

  return (
    <button
      type={type}
      style={baseStyle}
      onClick={onClick}
      disabled={disabled}
      className={className}
      onMouseEnter={() => !disabled && setIsHovered(true)}
      onMouseLeave={() => {
        !disabled && setIsHovered(false);
        !disabled && setIsActive(false);
      }}
      onMouseDown={() => !disabled && setIsActive(true)}
      onMouseUp={() => !disabled && setIsActive(false)}
    >
      {icon && iconPosition === "left" && <span style={iconStyle}>{icon}</span>}
      {children}
      {icon && iconPosition === "right" && (
        <span style={iconStyle}>{icon}</span>
      )}
    </button>
  );
};

export default Button;
