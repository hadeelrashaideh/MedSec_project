import React, { createContext, useContext, useState, useEffect } from "react";
import {
  initializeKeyExchange,
  completeKeyExchange,
  setupKeyRefresh,
} from "../services/cryptoService";

// Create auth context
const AuthContext = createContext();

// Auth provider component
export const AuthProvider = ({ children }) => {
  const [tokens, setTokens] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [encryptionStatus, setEncryptionStatus] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Helper to normalize role
  const normalizeRole = (role) => {
    if (!role) return "";
    // Convert any role format to lowercase "lab" or "doctor"
    const normalizedRole = role.toLowerCase().trim();
    console.log(`Normalized role: ${role} → ${normalizedRole}`);
    return normalizedRole;
  };

  // Initialize tokens from localStorage on component mount
  useEffect(() => {
    const accessToken = localStorage.getItem("accessToken");
    const refreshTkn = localStorage.getItem("refreshToken");

    if (accessToken && refreshTkn) {
      setTokens({
        access: accessToken,
        refresh: refreshTkn,
      });

      // Try to restore user data from localStorage
      const storedRole = localStorage.getItem("userRole");
      const storedUserId = localStorage.getItem("userId");
      const storedEmail = localStorage.getItem("userEmail");

      if (storedRole && storedUserId) {
        console.log("Restoring user session with role:", storedRole);
        setUser({
          username: storedEmail,
          email: storedEmail,
          role: normalizeRole(storedRole),
          id: storedUserId,
        });
        setIsAuthenticated(true);
      }

      // Check if encryption is established
      if (sessionStorage.getItem("encryption_established") === "true") {
        setEncryptionStatus(true);
      }
    }

    // Set up periodic key refresh
    const cleanupKeyRefresh = setupKeyRefresh(60); // Refresh every 60 minutes

    return () => {
      cleanupKeyRefresh();
    };
  }, []);

  // Login function
  const login = async (email, password) => {
    setLoading(true);
    setError(null);

    try {
      console.log("Sending login request with email:", email);
      const response = await fetch("http://localhost:8000/api/auth/login/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      console.log("Login response:", data);

      if (!response.ok) {
        throw new Error(data.detail || "Login failed");
      }

      // Validate role data
      if (!data.role) {
        console.error("No role received from API:", data);
        throw new Error("Invalid user data received: missing role");
      }

      const normalizedRole = normalizeRole(data.role);
      console.log(
        `User role from API: ${data.role} (normalized: ${normalizedRole})`
      );

      // Save tokens to state and localStorage
      setTokens({
        access: data.access,
        refresh: data.refresh,
      });

      localStorage.setItem("accessToken", data.access);
      localStorage.setItem("refreshToken", data.refresh);

      // Store user data in localStorage for persistence
      localStorage.setItem("userRole", normalizedRole);
      localStorage.setItem("userId", data.user_id);
      localStorage.setItem("userEmail", data.email || email);

      // Set user data in state
      setUser({
        username: data.username || data.email || email,
        email: data.email || email,
        role: normalizedRole,
        id: data.user_id,
      });

      // Update authentication state
      setIsAuthenticated(true);

      // Perform key exchange after successful login
      try {
        // Step 1: Initialize key exchange
        const dhParams = await initializeKeyExchange(data.access);

        // Step 2: Complete key exchange
        await completeKeyExchange(data.access, dhParams);

        // Update encryption status
        setEncryptionStatus(true);
      } catch (encryptionError) {
        console.warn(
          "Encryption key exchange failed, using default encryption:",
          encryptionError
        );
        setEncryptionStatus(false);
      }

      setLoading(false);
      return {
        ...data,
        role: normalizedRole,
      };
    } catch (err) {
      console.error("Login error:", err);
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("userRole");
    localStorage.removeItem("userId");
    localStorage.removeItem("userEmail");
    sessionStorage.removeItem("encryption_established");
    setTokens(null);
    setUser(null);
    setIsAuthenticated(false);
    setEncryptionStatus(false);
  };

  // Token refresh function
  const refreshToken = async () => {
    if (!tokens?.refresh) return false;

    try {
      const response = await fetch(
        "http://localhost:8000/api/auth/token/refresh/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ refresh: tokens.refresh }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        // Clear tokens if refresh fails
        logout();
        return false;
      }

      // Update tokens
      const newTokens = {
        ...tokens,
        access: data.access,
      };

      setTokens(newTokens);
      localStorage.setItem("accessToken", data.access);

      // Refresh encryption key
      try {
        const dhParams = await initializeKeyExchange(data.access);
        await completeKeyExchange(data.access, dhParams);
        setEncryptionStatus(true);
      } catch (encryptionError) {
        console.warn("Failed to refresh encryption key:", encryptionError);
      }

      return true;
    } catch (err) {
      console.error("Token refresh error:", err);
      logout();
      return false;
    }
  };

  const value = {
    tokens,
    user,
    loading,
    error,
    login,
    logout,
    refreshToken,
    encryptionStatus,
    isAuthenticated,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook for using auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
