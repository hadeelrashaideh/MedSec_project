import React, { createContext, useContext, useState } from "react";
import { useAuth } from "./AuthContext";

// Create the patient context
const PatientContext = createContext();

// Patient provider component
export const PatientProvider = ({ children }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(null);
  const { tokens, refreshToken, encryptionStatus } = useAuth();

  // Helper function to handle API requests with token refresh
  const apiRequest = async (url, method, body = null, isFormData = false) => {
    setLoading(true);
    setError(null);

    try {
      let headers = {
        Authorization: `Bearer ${tokens.access}`,
      };

      if (!isFormData) {
        headers["Content-Type"] = "application/json";
      }

      const requestOptions = {
        method,
        headers,
        body: isFormData ? body : body ? JSON.stringify(body) : null,
      };

      let response = await fetch(url, requestOptions);

      // If token expired, try to refresh it
      if (response.status === 401) {
        const refreshed = await refreshToken();
        if (refreshed) {
          // Update headers with new token
          headers.Authorization = `Bearer ${localStorage.getItem(
            "accessToken"
          )}`;
          // Retry the request
          response = await fetch(url, {
            ...requestOptions,
            headers,
          });
        }
      }

      const data = await response.json();
      console.log("data /////////////", data);
      setLoading(false);

      if (!response.ok) {
        setError(data.detail || "Operation failed");
        return { success: false, error: data.detail || "Operation failed" };
      }

      return { success: true, data };
    } catch (err) {
      setLoading(false);
      setError("Network error or server unavailable");
      return { success: false, error: "Network error" };
    }
  };

  // Add new patient
  const addPatient = async (patientData) => {
    const formData = new FormData();
    formData.append("id", patientData.id);
    formData.append("name", patientData.name);
    formData.append("age", patientData.age);

    // Handle notes field (optional)
    if (patientData.notes) {
      formData.append("note", patientData.notes);
    }

    // Handle image if available
    if (patientData.image) {
      formData.append("image", patientData.image);
    }

    // Add encryption flag if E2E encryption is established
    if (encryptionStatus) {
      formData.append("use_e2e_encryption", "true");
    }

    const result = await apiRequest(
      "http://localhost:8000/api/patients/",
      "POST",
      formData,
      true
    );

    return result.success ? result.data : null;
  };

  // Search for patient by ID
  const findPatientById = async (id) => {
    try {
      const result = await apiRequest(
        `http://localhost:8000/api/patients/${id}/`,
        "GET"
      );

      if (result.success) {
        console.log("result.data /////////////", result.data);
        return result.data;
      } else {
        console.error("Error finding patient:", result.error);
        return null;
      }
    } catch (err) {
      console.error("Exception in findPatientById:", err);
      return null;
    }
  };

  // Fetch image with authentication and Diffie-Hellman decryption
  const fetchAuthenticatedImage = async (url) => {
    setImageLoading(true);
    setImageError(null);

    try {
      // Add a cache-busting parameter to prevent browser caching
      const cacheParam = `?nocache=${new Date().getTime()}`;

      // Add E2E flag if encryption is established
      const e2eParam = encryptionStatus ? "&use_e2e=true" : "";

      const response = await fetch(`${url}${cacheParam}${e2eParam}`, {
        headers: {
          Authorization: `Bearer ${tokens.access}`,
        },
        // Ensure no caching
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(
          `Failed to load image: ${response.status} ${response.statusText}`
        );
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      setImageLoading(false);
      return objectUrl;
    } catch (error) {
      console.error("Error fetching image:", error);
      setImageError(error.message);
      setImageLoading(false);
      return null;
    }
  };

  const value = {
    loading,
    error,
    imageLoading,
    imageError,
    addPatient,
    findPatientById,
    fetchAuthenticatedImage,
    encryptionStatus,
  };

  return (
    <PatientContext.Provider value={value}>{children}</PatientContext.Provider>
  );
};

// Custom hook for using patient context
export const usePatients = () => {
  const context = useContext(PatientContext);
  if (!context) {
    throw new Error("usePatients must be used within a PatientProvider");
  }
  return context;
};
